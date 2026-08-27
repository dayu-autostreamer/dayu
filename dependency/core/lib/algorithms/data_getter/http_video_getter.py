import abc
import copy
import json
import math
import os
import time
from collections.abc import Mapping

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context, Counter, NameMaintainer
from core.lib.content import TaskIdentity
from .base_getter import BaseDataGetter, DataGetterStatus
from core.lib.network import http_request

__all__ = ('HttpVideoGetter',)


class _BurstArrivalShaper:
    """Reshape HTTP task arrivals without changing their long-term rate."""

    PARAMETER_NAME = 'HTTP_VIDEO_TASK_ARRIVAL_BURST'
    _FIELDS = frozenset((
        'tasks_per_burst',
        'intra_burst_rate_multiplier',
    ))

    def __init__(self, tasks_per_burst, intra_burst_rate_multiplier):
        self.tasks_per_burst = tasks_per_burst
        self.intra_burst_rate_multiplier = intra_burst_rate_multiplier
        self.reset()

    @classmethod
    def from_parameter(cls, value):
        if value is None or value == {}:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(
                f'{cls.PARAMETER_NAME} must be a mapping or omitted'
            )

        fields = set(value)
        missing = cls._FIELDS - fields
        unknown = fields - cls._FIELDS
        if missing or unknown:
            details = []
            if missing:
                details.append(f'missing fields: {sorted(missing)}')
            if unknown:
                details.append(f'unknown fields: {sorted(unknown)}')
            raise ValueError(
                f'invalid {cls.PARAMETER_NAME}: ' + '; '.join(details)
            )

        tasks_per_burst = value['tasks_per_burst']
        if (
            isinstance(tasks_per_burst, bool)
            or not isinstance(tasks_per_burst, int)
            or tasks_per_burst < 2
        ):
            raise ValueError(
                f'{cls.PARAMETER_NAME}.tasks_per_burst must be an integer >= 2'
            )

        rate_multiplier = value['intra_burst_rate_multiplier']
        if isinstance(rate_multiplier, bool):
            raise ValueError(
                f'{cls.PARAMETER_NAME}.intra_burst_rate_multiplier '
                'must be finite and > 1'
            )
        try:
            rate_multiplier = float(rate_multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'{cls.PARAMETER_NAME}.intra_burst_rate_multiplier '
                'must be finite and > 1'
            ) from exc
        if not math.isfinite(rate_multiplier) or rate_multiplier <= 1.0:
            raise ValueError(
                f'{cls.PARAMETER_NAME}.intra_burst_rate_multiplier '
                'must be finite and > 1'
            )
        return cls(tasks_per_burst, rate_multiplier)

    def reset(self):
        self._task_index = 0
        self._natural_interval_budget_s = 0.0
        self._shaped_interval_budget_s = 0.0

    def next_interval(self, natural_interval_s):
        self._natural_interval_budget_s += natural_interval_s
        # Compress the first N-1 gaps, then repay the exact accumulated
        # source-time budget at the cycle boundary.
        if self._task_index < self.tasks_per_burst - 1:
            interval_s = (
                natural_interval_s / self.intra_burst_rate_multiplier
            )
            self._shaped_interval_budget_s += interval_s
            self._task_index += 1
            return interval_s

        interval_s = (
            self._natural_interval_budget_s
            - self._shaped_interval_budget_s
        )
        self.reset()
        return interval_s


@ClassFactory.register(ClassType.GEN_GETTER, alias='http_video')
class HttpVideoGetter(BaseDataGetter, abc.ABC):
    """
    get video data from http (fastapi)
    preprocessed video data with accuracy information
    """

    def __init__(self):
        self.file_name = None
        self.hash_codes = None

        self.file_suffix = 'mp4'

        self._arrival_burst = _BurstArrivalShaper.from_parameter(
            Context.get_parameter(
                _BurstArrivalShaper.PARAMETER_NAME,
                None,
                direct=False,
            )
        )
        self._next_arrival_monotonic = None
        self._exhausted_instance_id = ''
        self._exhausted_endpoint_went_down = False

    def _source_status(self, system):
        response = http_request(
            system.video_data_source + '/status',
            method='GET',
            timeout=1.0,
            retry=1,
        )
        return response if isinstance(response, dict) else None

    def mark_source_exhausted(self, system):
        status = self._source_status(system)
        self._exhausted_instance_id = str(
            (status or {}).get('instance_id') or ''
        )
        self._exhausted_endpoint_went_down = status is None

    def datasource_reset_ready(self, system):
        """Return true only after the finite source process was recreated."""

        status = self._source_status(system)
        if status is None:
            self._exhausted_endpoint_went_down = True
            return False
        instance_id = str(status.get('instance_id') or '')
        changed = bool(
            instance_id
            and self._exhausted_instance_id
            and instance_id != self._exhausted_instance_id
        )
        restarted_after_down = (
            self._exhausted_endpoint_went_down
            and not bool(status.get('exhausted'))
        )
        if not changed and not restarted_after_down:
            return False
        self._exhausted_instance_id = ''
        self._exhausted_endpoint_went_down = False
        self._reset_arrival_clock()
        return True

    def request_source_data(self, system, task_id):
        data = {
            'source_id': system.source_id,
            'task_id': task_id,
            'meta_data': copy.deepcopy(system.meta_data),
            'raw_meta_data': copy.deepcopy(system.raw_meta_data),
            'gen_filter_name': Context.get_parameter('GEN_FILTER_NAME'),
            'gen_process_name': Context.get_parameter('GEN_PROCESS_NAME'),
            'gen_compress_name': Context.get_parameter('GEN_COMPRESS_NAME')
        }

        response = None
        self.hash_codes = None
        while not self.hash_codes or not response:
            self.hash_codes = http_request(system.video_data_source + '/source', method='GET',
                                           data={'data': json.dumps(data)})
            if self.hash_codes == []:
                return False

            if self.hash_codes:
                shared = http_request(
                    system.video_data_source + '/shared_file',
                    method='GET',
                    timeout=1.0,
                    retry=1,
                )
                shared_value = (
                    shared.get('file_name')
                    if isinstance(shared, dict) else None
                )
                shared_file = str(shared_value or '')
                if shared_file and os.path.isfile(shared_file):
                    self.file_name = shared_file
                    return True
                response = http_request(
                    system.video_data_source + '/file',
                    method='GET',
                    no_decode=True,
                )
            else:
                time.sleep(1)

        self.file_name = NameMaintainer.get_task_data_file_name(system.source_id, task_id, self.file_suffix)

        with open(self.file_name, 'wb') as f:
            f.write(response.content)
        return True

    def _reset_arrival_clock(self):
        self._next_arrival_monotonic = None
        if self._arrival_burst is not None:
            self._arrival_burst.reset()

    def _next_arrival_interval(self, system, actual_buffer_size):
        buffer_size = (
            actual_buffer_size
            if actual_buffer_size is not None
            else system.meta_data['buffer_size']
        )
        natural_interval_s = buffer_size / system.meta_data['fps']
        if self._arrival_burst is None:
            return natural_interval_s
        return self._arrival_burst.next_interval(natural_interval_s)

    def _wait_until_next_arrival(
        self,
        system,
        actual_buffer_size,
        current_offer_monotonic,
    ):
        """Pace the next HTTP task offer after the current task is emitted."""

        interval = self._next_arrival_interval(system, actual_buffer_size)
        if self._next_arrival_monotonic is None:
            self._next_arrival_monotonic = current_offer_monotonic
        self._next_arrival_monotonic += interval
        sleep_time = max(
            0.0,
            self._next_arrival_monotonic - time.monotonic(),
        )
        LOGGER.info(
            f'[Camera Simulation] source {system.source_id}: '
            f'next task offer in {sleep_time}s'
        )
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    def __call__(self, system, task_identity=None):
        # The call begins immediately after the Generator has obtained the
        # scheduling decision for this logical task offer.  Keep this local
        # anchor solely to pace the next offer; it never enters Task metadata
        # or a scheduling request.
        current_offer_monotonic = time.monotonic()
        if task_identity is None:
            task_identity = TaskIdentity.create(system.source_id, Counter.get_count('task_id'))
        new_task_id = task_identity.task_id

        source_ready = self.request_source_data(system, new_task_id)
        if not source_ready:
            LOGGER.info(f'[Camera Simulation] source {system.source_id}: datasource exhausted, skip current round')
            self.mark_source_exhausted(system)
            time.sleep(1)
            return DataGetterStatus.EXHAUSTED

        try:
            actual_buffer_size = len(self.hash_codes) if self.hash_codes else 0
            system.cumulative_scheduling_frame_count += (
                actual_buffer_size *
                system.raw_meta_data.get('fps', 0) /
                system.meta_data.get('fps', 1)
            )

            new_task = system.generate_task(new_task_id, copy.deepcopy(system.task_dag),
                                            copy.deepcopy(system.service_deployment),
                                            copy.deepcopy(system.meta_data),
                                            self.file_name, self.hash_codes,
                                            task_identity=task_identity)
            submitted = system.submit_task_to_controller(new_task)
        finally:
            FileOps.remove_file(self.file_name)

        # A materialized generation attempt advances the HTTP replay clock even
        # if downstream admission rejects it.  Waiting after cleanup keeps the
        # current task's submission and SLO independent of the next arrival,
        # while the synchronous getter return gates the next scheduling round.
        self._wait_until_next_arrival(
            system,
            actual_buffer_size,
            current_offer_monotonic,
        )
        return submitted
