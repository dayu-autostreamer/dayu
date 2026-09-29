import copy
import json
import gzip
import shutil
import tempfile
import threading
import re
import uuid
from collections import deque
from dataclasses import dataclass

import os
import time
from core.lib.content import Task
from core.lib.common import LOGGER, Context, YamlOps, FileOps, Counter, Queue, TaskConstant, \
    ConfigBoundInstanceCache
from core.lib.network import connection_host, http_request, NetworkAPIPath, NetworkAPIMethod
from core.lib.estimation import Timer

from runtime_orchestrator import (
    RuntimeOperationCancelled,
    RuntimeOrchestrator,
    RuntimeRetirementPending,
)
from runtime_telemetry import RuntimeTelemetryCache
from template_helper import TemplateHelper


def _indent_json_block(text, prefix='    '):
    return '\n'.join(f'{prefix}{line}' if line else prefix for line in text.splitlines())


@dataclass
class _InstallAdmission:
    install_id: str
    cancel_event: threading.Event
    done_event: threading.Event
    phase: str = 'preparing-install'
    operation_id: str = ''

    def cancel(self):
        if self.phase != 'cancelling-install':
            self.phase = 'cancelling-install'
            self.operation_id = str(uuid.uuid4())
        self.cancel_event.set()


@dataclass
class _StopAdmission:
    install_id: str
    done_event: threading.Event
    operation_id: str
    result: object = None


class BackendCore:
    _RESULT_REQUEST_TIMEOUT_SECONDS = 5.0
    _RESULT_WINDOW_SIZE = 20

    def __init__(self):

        self.template_helper = TemplateHelper(Context.get_default_file_path())

        self.namespace = ''
        self.image_meta = None
        self.schedulers = None
        self.services = None

        self.result_visualization_configs = None
        self.system_visualization_configs = None
        self.customized_source_result_visualization_configs = {}
        self.result_visualization_cache = ConfigBoundInstanceCache(
            factory=lambda vf: Context.get_algorithm(
                'RESULT_VISUALIZER',
                al_name=vf['hook_name'],
                **(dict(eval(vf['hook_params'])) if 'hook_params' in vf else {}),
                variables=vf['variables']
            )
        )
        self.system_visualization_cache = ConfigBoundInstanceCache(
            factory=lambda vf: Context.get_algorithm(
                'SYSTEM_VISUALIZER',
                al_name=vf['hook_name'],
                **(dict(eval(vf['hook_params'])) if 'hook_params' in vf else {}),
                variables=vf['variables']
            )
        )

        self.parse_base_info()

        self.source_configs = []

        self.dags = []

        self.time_ticket = 0

        self.result_url = None
        self.result_file_url = None
        self.resource_url = None
        self.log_fetch_url = None

        self.runtime_telemetry = RuntimeTelemetryCache(
            request=lambda *args, **kwargs: http_request(*args, **kwargs),
            runtime_metrics=lambda refs, request_timeout_seconds: (
                self.runtime_orchestrator.sample_runtime_metrics(
                    refs,
                    request_timeout_seconds=request_timeout_seconds,
                )
            ),
        )

        self.inner_datasource = self.check_simulation_datasource()
        self.source_open = False
        self.source_label = ''
        self.query_lock = threading.Lock()
        self._query_generation = 0
        self._query_cancel_event = None
        self._result_thread = None
        # Query admission follows the committed runtime lifecycle.  It is
        # enabled only after install/recovery publishes an active directory
        # and is disabled atomically with query cancellation before uninstall.
        self._query_admission_enabled = False
        self.task_results = {}

        self.is_get_result = False
        # Lifecycle cancellation is coordinated independently from the
        # RuntimeOrchestrator transaction lock.  An uninstall request registers
        # here first and can therefore signal an install that is blocked in a
        # RuntimeService watch before waiting for the serialized cleanup
        # transaction. A single-flight stop admission also closes the
        # stop-before-token race: no install can register before durable stop
        # acceptance, and concurrent callers observe the same result.
        self._lifecycle_control_lock = threading.Lock()
        self._closed = False
        self._install_admission = None
        self._stop_admission = None
        self._bound_runtime_key = None
        self._local_runtime_error_key = None
        self._local_runtime_error = ''
        self._runtime_reconcile_lock = threading.Lock()
        self._runtime_reconcile_stop_event = None
        self._runtime_reconcile_thread = None
        self._runtime_recovery_stop_event = threading.Event()
        self._runtime_recovery_wake_event = threading.Event()
        self._runtime_recovery_requested = False
        self._runtime_recovery_thread = None
        self.runtime_orchestrator = RuntimeOrchestrator(
            self.template_helper,
            self.namespace,
        )
        redeploy_interval = Context.get_parameter('REDEPLOYMENT_REQUEST_INTERVAL', default=20, direct=False)
        self.processor_redeployment_interval_s = max(0.0, float(redeploy_interval))
        self.system_log_store_path = 'system_log_store.jsonl'
        self.system_log_lock = threading.Lock()
        self.system_log_retention_records = max(
            0,
            int(Context.get_parameter('SYSTEM_LOG_RETENTION_RECORDS', 0, direct=False))
        )
        self.system_log_compact_interval = max(
            1,
            int(Context.get_parameter('SYSTEM_LOG_COMPACT_INTERVAL', 200, direct=False))
        )
        self.system_log_record_count = self._count_jsonl_records(self.system_log_store_path)

        self.default_visualization_image = 'default_visualization.png'

    def parse_base_info(self):
        try:
            base_info = self.template_helper.load_base_info()
            self.namespace = base_info['namespace']
            self.image_meta = base_info['default-image-meta']
            self.schedulers = base_info['scheduler-policies']
            self.services = base_info['services']
            self.result_visualization_configs = base_info['result-visualizations']
            self.system_visualization_configs = base_info['system-visualizations']
        except KeyError as e:
            LOGGER.warning(f'Parse base info failed: {str(e)}')

    @staticmethod
    def _runtime_key(directory):
        if directory is None:
            return None
        return str(directory.install_id), int(directory.revision)

    def _activate_local_runtime(self, directory, start_reconcile=False):
        """Publish one directory to Backend-local readers after durable commit."""
        if self._closed:
            raise RuntimeError('backend lifecycle is closed')
        runtime_key = self._runtime_key(directory)
        reconcile_started = False
        try:
            self._bind_runtime_urls(directory)
            self.runtime_telemetry.start()
            if start_reconcile:
                self._start_runtime_reconcile_loop(directory.install_id)
                reconcile_started = True
            # Publish the generation before opening query admission. Production
            # callers hold lifecycle control across this method, so management
            # readers cannot observe the intermediate value; query callers only
            # become admissible after every local projection field is complete.
            self._bound_runtime_key = runtime_key
            self._local_runtime_error_key = None
            self._local_runtime_error = ''
            with self.query_lock:
                self._query_admission_enabled = True
        except Exception as exc:
            self._bound_runtime_key = None
            self._local_runtime_error_key = runtime_key
            self._local_runtime_error = f'local runtime activation failed: {exc}'
            try:
                if reconcile_started:
                    self._stop_runtime_reconcile_loop()
            except Exception:
                LOGGER.exception('failed to stop reconcile after local activation error')
            try:
                with self.query_lock:
                    self._query_admission_enabled = False
                    self._close_query_locked()
            except Exception:
                LOGGER.exception('failed to close query after local activation error')
            try:
                self.runtime_telemetry.unbind()
            except Exception:
                LOGGER.exception('failed to unbind telemetry after local activation error')
            self.resource_url = None
            self.result_url = None
            self.result_file_url = None
            self.log_fetch_url = None
            raise

    def _ensure_local_runtime_projection(self, session, cancel_event=None):
        """Repair a missing local projection from the in-memory directory."""
        if session is None or session.phase != 'active':
            return False
        session_key = (
            session.install_id,
            int(session.active_directory_revision),
        )
        if self._bound_runtime_key == session_key:
            return False
        with self._lifecycle_control_lock:
            if (
                    self._stop_admission is not None
                    or (cancel_event is not None and cancel_event.is_set())):
                return False
            if self._bound_runtime_key == session_key:
                return False
            directory = self.runtime_orchestrator.active_directory()
            if self._runtime_key(directory) != session_key:
                raise RuntimeError(
                    'active RuntimeDirectory does not match the durable session generation'
                )
            # active_directory() is a process snapshot lookup. Retrying this
            # projection performs no Kubernetes discovery or list call.
            self._activate_local_runtime(directory)
            return True

    def _recover_runtime_session(self, stop_event=None):
        """Perform one recovery attempt; return whether it reached a stable state."""
        try:
            session = self.runtime_orchestrator.recover()
            if stop_event is not None and stop_event.is_set():
                return True
            if session is None:
                return True
            if session.phase in {'uninstalling', 'finalizing-uninstall'}:
                # A durable uninstall intent must resume before this process
                # can re-open query admission. The same lifecycle worker
                # repeats each exact-UID boundary idempotently without blocking
                # backend startup or management reads.
                with self._lifecycle_control_lock:
                    if stop_event is not None and stop_event.is_set():
                        return True
                    self._start_runtime_reconcile_loop(session.install_id)
                return True
            if session.phase != 'active':
                LOGGER.warning(
                    f'[Runtime Recovery] Session {session.install_id} requires operator cleanup: '
                    f'phase={session.phase}, error={session.last_error}'
                )
                return True
            directory = self.runtime_orchestrator.active_directory()
            if directory is None:
                raise RuntimeError('recovered active session has no RuntimeDirectory')
            with self._lifecycle_control_lock:
                if stop_event is not None and stop_event.is_set():
                    return True
                stop = self._stop_admission
                if stop is not None and stop.install_id in {'', session.install_id}:
                    return True
                current = self.runtime_orchestrator.current_session()
                if (
                        current is None
                        or current.install_id != session.install_id
                        or current.phase != 'active'):
                    return False
                self._activate_local_runtime(directory, start_reconcile=True)
            return True
        except Exception as exc:
            # Keep the management API available for inspection/uninstall even
            # when an external dependency is unavailable during process start.
            try:
                current = self.runtime_orchestrator.current_session()
            except Exception:
                current = None
            if current is not None and current.phase == 'active':
                runtime_key = (
                    current.install_id,
                    int(current.active_directory_revision),
                )
                with self._lifecycle_control_lock:
                    self._bound_runtime_key = None
                    if self._local_runtime_error_key != runtime_key:
                        self._local_runtime_error_key = runtime_key
                        self._local_runtime_error = (
                            f'local runtime recovery failed: {exc}'
                        )
            LOGGER.warning(f'[Runtime Recovery] Managed runtime recovery failed: {exc}')
            LOGGER.exception(exc)
            return False

    def _start_runtime_recovery_async(self):
        with self._lifecycle_control_lock:
            if self._closed:
                return
            if self._runtime_recovery_thread is not None:
                # Do not lose a trigger against a worker that has just reached
                # a stable snapshot but has not yet cleared its ownership.
                self._runtime_recovery_requested = True
                self._runtime_recovery_wake_event.set()
                return
            self._runtime_recovery_stop_event.clear()
            self._runtime_recovery_wake_event.clear()
            self._runtime_recovery_requested = False
            thread = threading.Thread(
                target=self.run_runtime_recovery,
                args=(self._runtime_recovery_stop_event,),
                name='dayu-runtime-recovery',
                daemon=True,
            )
            self._runtime_recovery_thread = thread
            try:
                thread.start()
            except Exception:
                self._runtime_recovery_thread = None
                self._runtime_recovery_stop_event.set()
                raise

    def start(self):
        """Start lifecycle workers under the owning application's lifespan."""
        with self._lifecycle_control_lock:
            self._closed = False
        if os.getenv('DAYU_RUNTIME_CONTROL_PLANE', '').lower() == 'true':
            self._start_runtime_recovery_async()

    def run_runtime_recovery(self, stop_event):
        retry_delay = 1.0
        try:
            while not stop_event.is_set():
                if self._recover_runtime_session(stop_event=stop_event):
                    with self._lifecycle_control_lock:
                        if stop_event.is_set():
                            if self._runtime_recovery_thread is threading.current_thread():
                                self._runtime_recovery_thread = None
                            self._runtime_recovery_requested = False
                            return
                        if self._runtime_recovery_requested:
                            self._runtime_recovery_requested = False
                            self._runtime_recovery_wake_event.clear()
                            retry_delay = 1.0
                            continue
                        if self._runtime_recovery_thread is threading.current_thread():
                            self._runtime_recovery_thread = None
                        return
                woke = self._runtime_recovery_wake_event.wait(retry_delay)
                self._runtime_recovery_wake_event.clear()
                if stop_event.is_set():
                    return
                if woke:
                    with self._lifecycle_control_lock:
                        self._runtime_recovery_requested = False
                    retry_delay = 1.0
                else:
                    retry_delay = min(retry_delay * 2.0, 30.0)
        finally:
            with self._lifecycle_control_lock:
                if self._runtime_recovery_thread is threading.current_thread():
                    self._runtime_recovery_thread = None

    def get_log_file_name(self):
        base_info = self.template_helper.load_base_info()
        load_file_name = base_info['log-file-name']
        if not load_file_name:
            return None
        return load_file_name.split('.')[0]

    def parse_and_apply_templates(
            self, policy, source_deploy, source_label='', install_id=''):
        """Install one transactional managed-runtime session."""
        install_id = str(install_id or '').strip()
        try:
            canonical_install_id = str(uuid.UUID(install_id))
        except (ValueError, AttributeError, TypeError):
            return False, 'install_id must be a canonical UUID'
        if install_id != canonical_install_id:
            return False, 'install_id must be a canonical UUID'

        cancel_event = threading.Event()
        # Complete the one-off ConfigMap snapshot load without holding process
        # admission control. The second read below is memory-only, so a slow API
        # server cannot freeze stop registration or lifecycle status sampling.
        self.runtime_orchestrator.current_session()
        with self._lifecycle_control_lock:
            if self._closed:
                return False, 'Backend lifecycle is closed'
            if self._stop_admission is not None:
                return False, 'Install cancelled by lifecycle operation'
            session = self.runtime_orchestrator.current_session()
            if session is not None:
                if session.phase in {'uninstalling', 'finalizing-uninstall'}:
                    return False, 'Uninstall is in progress'
                return False, 'A managed runtime session already exists; uninstall it before installing'
            if self._install_admission is not None:
                return False, 'Another install operation is already in progress'
            admission = _InstallAdmission(
                install_id=install_id,
                cancel_event=cancel_event,
                done_event=threading.Event(),
                operation_id=str(uuid.uuid4()),
            )
            self._install_admission = admission
            self._local_runtime_error_key = None
            self._local_runtime_error = ''

        try:
            try:
                directory = self.runtime_orchestrator.install(
                    policy=policy,
                    source_deploy=source_deploy,
                    source_label=source_label,
                    install_id=install_id,
                    cancel_event=cancel_event,
                )
            except RuntimeOperationCancelled:
                return False, 'Install cancelled by lifecycle operation'
            except Exception as exc:
                LOGGER.warning(f'Managed runtime install failed: {exc}')
                LOGGER.exception(exc)
                recovery_required = False
                try:
                    current = self.runtime_orchestrator.current_session()
                    if (
                            current is not None
                            and current.install_id == install_id
                            and self.runtime_orchestrator.requires_recovery(current)):
                        recovery_required = True
                except Exception:
                    # Snapshot calibration is itself unavailable. A recovery
                    # controller is the only bounded way to determine whether
                    # the initial Session CAS committed before the lost reply.
                    recovery_required = True
                    LOGGER.exception(
                        'failed to inspect runtime session after install error'
                    )
                try:
                    if recovery_required:
                        self._start_runtime_recovery_async()
                except Exception:
                    LOGGER.exception(
                        'failed to start runtime recovery controller'
                    )
                return False, str(exc)

            # Publish the local projection under the same short admission
            # boundary used by stop registration. A stop either observes and
            # removes this projection, or cancels before it can become ready.
            with self._lifecycle_control_lock:
                if cancel_event.is_set() or self._stop_admission is not None:
                    return False, 'Install cancelled by lifecycle operation'
                session = self.runtime_orchestrator.current_session()
                if (
                        session is None
                        or session.install_id != directory.install_id
                        or session.phase != 'active'):
                    raise RuntimeError(
                        'install completed without an active RuntimeSession snapshot'
                    )
                self._activate_local_runtime(directory, start_reconcile=True)
            return True, 'Install services successfully'
        except Exception as exc:
            LOGGER.warning(f'Managed runtime local activation failed: {exc}')
            LOGGER.exception(exc)
            try:
                current = self.runtime_orchestrator.current_session()
                if (
                        current is not None
                        and current.install_id == install_id
                        and current.phase == 'active'):
                    self._start_runtime_recovery_async()
            except Exception:
                LOGGER.exception('failed to start local projection recovery controller')
            return False, str(exc)
        finally:
            with self._lifecycle_control_lock:
                if self._install_admission is admission:
                    self._install_admission = None
                admission.done_event.set()

    def parse_and_delete_templates(self, expected_install_id=''):
        """Persist stop intent and start managed runtime cleanup."""
        expected_install_id = str(expected_install_id or '').strip()
        if expected_install_id:
            try:
                canonical_install_id = str(uuid.UUID(expected_install_id))
            except (ValueError, AttributeError, TypeError):
                return False, 'install_id must be a canonical UUID'
            if expected_install_id != canonical_install_id:
                return False, 'install_id must be a canonical UUID'
        with self._lifecycle_control_lock:
            if self._closed:
                return False, 'Backend lifecycle is closed'
        # Complete the one-off Session load before admission serialization. The
        # second read below is memory-only and closes the install/stop race.
        self.runtime_orchestrator.current_session()
        # Register stop before touching any other lifecycle state.  This both
        # interrupts an in-flight install and prevents an overlapping install
        # from registering a token until this stop request has settled.
        with self._lifecycle_control_lock:
            if self._closed:
                return False, 'Backend lifecycle is closed'
            session = self.runtime_orchestrator.current_session()
            admission = self._install_admission
            pending_install_id = admission.install_id if admission is not None else ''
            if (
                    expected_install_id
                    and not (
                        (session is not None and session.install_id == expected_install_id)
                        or pending_install_id == expected_install_id
                    )):
                return True, 'Target installation is already absent'
            if admission is not None:
                admission.cancel()
            stop_admission = self._stop_admission
            if stop_admission is None:
                target_install_id = expected_install_id or (
                    session.install_id if session is not None else pending_install_id
                )
                stop_admission = _StopAdmission(
                    install_id=target_install_id,
                    done_event=threading.Event(),
                    operation_id=str(uuid.uuid4()),
                )
                self._stop_admission = stop_admission
                # Stop admission and query admission share one linearization
                # boundary. A client that observes preparing-uninstall can no
                # longer open a datasource generation, and the previous result
                # collector has already been fenced.
                try:
                    with self.query_lock:
                        self._query_admission_enabled = False
                        self._close_query_locked()
                except Exception as exc:
                    result = (False, str(exc))
                    stop_admission.result = result
                    if self._stop_admission is stop_admission:
                        self._stop_admission = None
                    stop_admission.done_event.set()
                    LOGGER.warning(f'Managed runtime query shutdown failed: {exc}')
                    LOGGER.exception(exc)
                    return result
                self._bound_runtime_key = None
                leader = True
            else:
                leader = False

        # Every caller observes the same durable acceptance result. In
        # particular, a concurrent follower cannot report success while the
        # leader has not yet persisted uninstall intent (or its failure).
        if not leader:
            stop_admission.done_event.wait()
            return stop_admission.result or (
                False, 'Uninstall admission ended without a result',
            )

        # If this stop raced an install before its first Session CAS, wait for
        # the cancelled install to release its token. It can no longer publish
        # locally because this stop admission remains registered, and any
        # exact resource identities it persisted are then owned by uninstall.
        if admission is not None:
            admission.done_event.wait()

        result = (False, 'Uninstall did not reach durable acceptance')
        try:
            session = self.runtime_orchestrator.current_session()
            uninstall_started = session is not None and session.phase in {
                'uninstalling', 'finalizing-uninstall',
            }
            if uninstall_started:
                self.runtime_telemetry.unbind()
                self.resource_url = None
                self.result_url = None
                self.result_file_url = None
                self.log_fetch_url = None
                self._ensure_runtime_reconcile_loop(session.install_id)
                result = (True, 'Uninstall services started')
            else:
                # Stop every producer of Scheduler/Kubernetes traffic before
                # the serialized uninstall transaction. A failed uninstall
                # leaves telemetry and task admission deliberately unbound.
                self._stop_runtime_reconcile_loop()
                self.runtime_telemetry.unbind()
                session = self.runtime_orchestrator.begin_uninstall(
                    stop_admission.install_id,
                )
                self.resource_url = None
                self.result_url = None
                self.result_file_url = None
                self.log_fetch_url = None
                if session is not None:
                    self._local_runtime_error_key = None
                    self._local_runtime_error = ''
                    self._start_runtime_reconcile_loop(session.install_id)
                    result = (True, 'Uninstall services started')
                else:
                    self._local_runtime_error_key = None
                    self._local_runtime_error = ''
                    result = (True, 'No managed services are installed')
        except Exception as exc:
            LOGGER.warning(f'Managed runtime uninstall failed: {exc}')
            LOGGER.exception(exc)
            try:
                current = self.runtime_orchestrator.current_session()
            except Exception:
                current = None
            if current is not None and current.phase == 'active':
                with self._lifecycle_control_lock:
                    self._local_runtime_error_key = (
                        current.install_id,
                        int(getattr(current, 'active_directory_revision', 0) or 0),
                    )
                    self._local_runtime_error = f'local runtime shutdown failed: {exc}'
            result = (False, str(exc))
        finally:
            with self._lifecycle_control_lock:
                stop_admission.result = result
                if self._stop_admission is stop_admission:
                    self._stop_admission = None
                stop_admission.done_event.set()
        return result

    def parse_and_redeploy_services(self, policy=None, cancel_event=None):
        """Publish a processor rollout; unchanged plans are a successful no-op."""
        with self._lifecycle_control_lock:
            if self._closed:
                return False, 'Backend lifecycle is closed'
        session = self.runtime_orchestrator.current_session()
        if session is None:
            return False, 'no managed runtime session exists'
        policy = policy or self.find_scheduler_policy_by_id(session.policy_id)
        if policy is None:
            return False, f'scheduler policy {session.policy_id!r} does not exist'
        try:
            changed = self.runtime_orchestrator.redeploy(
                policy,
                cancel_event=cancel_event,
            )
        except RuntimeOperationCancelled:
            return False, 'Redeployment cancelled by lifecycle operation'
        except RuntimeRetirementPending:
            return False, 'Redeployment deferred while the previous revision retires'
        except Exception as exc:
            LOGGER.warning(f'Managed processor rollout failed: {exc}')
            LOGGER.exception(exc)
            return False, str(exc)
        if changed:
            directory = self.runtime_orchestrator.active_directory()
            try:
                with self._lifecycle_control_lock:
                    if (
                        self._stop_admission is not None
                        or (cancel_event is not None and cancel_event.is_set())
                    ):
                        return False, 'Redeployment cancelled by lifecycle operation'
                    if directory is None:
                        raise RuntimeError(
                            'redeployment committed without an active RuntimeDirectory'
                        )
                    self._activate_local_runtime(directory)
            except Exception as exc:
                LOGGER.warning(f'Managed runtime local projection failed: {exc}')
                LOGGER.exception(exc)
                return False, str(exc)
        return True, 'Redeployment succeeded' if changed else 'Deployment is unchanged'

    def find_service_by_id(self, service_id):
        for service in self.services:
            if service['id'] == service_id:
                return service
        return None

    @staticmethod
    def service_io_labels(service, field):
        service_id = service.get('id') or service.get('service') or '<unknown>'
        value = service.get(field)
        if not isinstance(value, list):
            return None, f"Service '{service_id}' field '{field}' must be a list of type labels"
        if any(not isinstance(item, str) or not item for item in value):
            return None, f"Service '{service_id}' field '{field}' must contain non-empty string labels"
        return value, None

    @classmethod
    def service_io_compatible(cls, parent_service, child_service):
        parent_outputs, error_msg = cls.service_io_labels(parent_service, 'output')
        if error_msg:
            return False, error_msg
        child_inputs, error_msg = cls.service_io_labels(child_service, 'input')
        if error_msg:
            return False, error_msg
        return bool(set(parent_outputs) & set(child_inputs)), None

    def find_dag_by_id(self, dag_id):
        for dag in self.dags:
            if dag['dag_id'] == dag_id:
                return dag['dag']
        return None

    def find_scheduler_policy_by_id(self, policy_id):
        for policy in self.schedulers:
            if policy['id'] == policy_id:
                return policy
        return None

    def find_datasource_configuration_by_label(self, source_label):
        for source_config in self.source_configs:
            if source_config['source_label'] == source_label:
                return source_config
        return None

    def fill_datasource_config(self, config):
        config['source_label'] = f'source_config_{Counter.get_count("source_label")}'
        source_list = config['source_list']
        for index, source in enumerate(source_list):
            source['id'] = index
            source['url'] = self.fill_datasource_url(source['url'], config['source_type'], config['source_mode'], index)

        config['source_list'] = source_list
        return config

    def fill_datasource_url(self, url, source_type, source_mode, source_id):
        if not self.inner_datasource:
            return url
        source_protocol = source_mode.split('_')[0]
        datasource_fqdn = connection_host(f'datasource-edge.{self.namespace}.svc.cluster.local')
        return f'{source_protocol}://{datasource_fqdn}:8000/{source_type}{source_id}'

    def get_edge_nodes(self):
        def sort_key(item):
            name = item['name']
            patterns = [
                (r'^edge(\d+)$', 0),
                (r'^edgexn(\d+)$', 1),
                (r'^edgex(\d+)$', 2),
                (r'^edgen(\d+)$', 3),
            ]
            for pattern, group in patterns:
                match = re.match(pattern, name)
                if match:
                    num = int(match.group(1))
                    return group, num
            return len(patterns), 0

        inventory = self.runtime_orchestrator.node_inventory()
        edge_nodes = [
            {'name': node_name}
            for node_name, record in inventory.items()
            if record.get('role') == 'edge' and record.get('ready')
        ]
        edge_nodes.sort(key=sort_key)
        return edge_nodes

    def management_lifecycle_snapshot(self):
        """Read admission and Session state without blocking the event loop."""
        # Complete the one-off durable load before taking admission control.
        self.runtime_orchestrator.current_session()
        with self._lifecycle_control_lock:
            admission = self._install_admission
            pending = (
                {
                    'kind': 'install',
                    'install_id': admission.install_id,
                    'phase': admission.phase,
                    'operation_id': admission.operation_id,
                }
                if admission is not None else None
            )
            # This second read is memory-only. Sampling both values under the
            # same admission boundary prevents combining an old Session with a
            # newer installation token from another client.
            session = self.runtime_orchestrator.current_session()
            stop = self._stop_admission
            if pending is None and stop is not None:
                pending = {
                    'kind': 'stop',
                    'install_id': stop.install_id,
                    'phase': 'preparing-uninstall',
                    'operation_id': stop.operation_id,
                }
            session_key = None
            if session is not None and session.phase == 'active':
                session_key = (
                    session.install_id,
                    int(getattr(session, 'active_directory_revision', 0) or 0),
                )
            stop_matches = bool(
                session is not None
                and stop is not None
                and stop.install_id in {'', session.install_id}
            )
            local_ready = bool(
                session_key is not None
                and self._bound_runtime_key == session_key
                and not stop_matches
                and not (
                    admission is not None
                    and admission.install_id == session.install_id
                )
            )
            local_error = (
                self._local_runtime_error
                if session_key == self._local_runtime_error_key else ''
            )
        return session, pending, local_ready, local_error

    def check_simulation_datasource(self):
        return bool(self.template_helper.load_base_info().get('datasource', {}).get('use-simulation'))

    def check_dag(self, dag):

        def topo_sort(graph):
            for node, node_info in graph.items():
                if node == TaskConstant.START.value:
                    continue
                service = self.find_service_by_id(node_info['id'])
                if not service:
                    error_msg = f"Missing service definition for node {node}"
                    LOGGER.error(f"DAG Validation Error: {error_msg}")
                    return False, error_msg
                for field in ('input', 'output'):
                    _, error_msg = self.service_io_labels(service, field)
                    if error_msg:
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg

            in_degree = {}
            for node in graph.keys():
                if node != TaskConstant.START.value:
                    in_degree[node] = len(graph[node]['prev'])
            queue = copy.deepcopy(graph[TaskConstant.START.value])
            topo_order = []

            while queue:
                parent = queue.pop(0)
                topo_order.append(parent)
                for child in graph[parent]['succ']:
                    parent_service = self.find_service_by_id(parent)
                    child_service = self.find_service_by_id(child)
                    if not parent_service or not child_service:
                        error_msg = f"Missing service definition for node {parent if not parent_service else child}"
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg
                    is_compatible, error_msg = self.service_io_compatible(parent_service, child_service)
                    if error_msg:
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg
                    if not is_compatible:
                        error_msg = (
                            f"Node connection mismatch, '{parent}' output '{parent_service['output']}', '{child}' input '{child_service['input']}' "
                        )
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg

                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

            if len(topo_order) != len(in_degree):
                error_msg = "DAG contains cycles or unreachable nodes"
                LOGGER.warning(f"DAG Validation Error: {error_msg}")
                return False, error_msg

            return True, "DAG validation passed"

        return topo_sort(dag.copy())

    def get_source_ids(self):
        source_ids = []
        source_config = self.find_datasource_configuration_by_label(self.source_label)
        if not source_config:
            return []
        for source in source_config['source_list']:
            source_ids.append(source['id'])

        return source_ids

    def prepare_result_visualization_data(self, task, is_last=False):
        source_id = task.get_source_id()
        viz_configs = self.customized_source_result_visualization_configs[source_id] \
            if source_id in self.customized_source_result_visualization_configs else self.result_visualization_configs
        viz_functions = self.result_visualization_cache.sync_and_get(viz_configs, namespace='result_visualizer')

        resource_snapshot = None
        if any(config.get('hook_name') == 'service_queue_length' for config in viz_configs):
            resource_snapshot = self.runtime_telemetry.snapshot()['resource']

        visualization_data = []
        for idx, (viz_config, viz_func) in enumerate(zip(viz_configs, viz_functions)):
            try:
                if 'save_expense' in viz_config and viz_config['save_expense'] and not is_last:
                    visualization_data.append({"id": idx, "data": {v: None for v in viz_config['variables']}})
                else:
                    if viz_config.get('hook_name') == 'service_queue_length':
                        data = viz_func(task, resource=resource_snapshot)
                    else:
                        data = viz_func(task)
                    visualization_data.append({"id": idx, "data": data})
            except Exception as e:
                LOGGER.warning(f'Failed to load result visualization data: {str(e)}')
                LOGGER.exception(e)

        return visualization_data

    def prepare_system_visualizations_data(self):
        viz_configs = self.system_visualization_configs
        viz_functions = self.system_visualization_cache.sync_and_get(viz_configs, namespace='system_visualizer')

        # Scheduler I/O is owned by one background sampler.  UI requests only
        # transform an immutable last-known-good snapshot.
        telemetry = self.runtime_telemetry.snapshot()
        resource_snapshot = telemetry['resource']
        scheduling_overhead = telemetry['scheduling_overhead']

        visualization_data = []
        for idx, (viz_config, viz_func) in enumerate(zip(viz_configs, viz_functions)):
            try:
                hook_name = viz_config.get('hook_name')
                if hook_name in {'cpu_usage', 'memory_usage'}:
                    data = viz_func(resource=resource_snapshot)
                elif hook_name == 'schedule_overhead':
                    data = viz_func(scheduling_overhead=scheduling_overhead)
                else:
                    data = viz_func()
                visualization_data.append({"id": idx, "data": data})
            except Exception as e:
                LOGGER.warning(f'Failed to load result visualization data: {str(e)}')
                LOGGER.exception(e)

        return visualization_data

    def parse_task_result(self, results, query_generation=None):
        for result in results:
            if result is None or result == '':
                continue

            task = Task.deserialize(result)

            source_id = task.get_source_id()
            LOGGER.debug(task.get_delay_info())

            task_copy = copy.deepcopy(task)
            with self.query_lock:
                if (
                    not self.source_open
                    or (
                        query_generation is not None
                        and query_generation != self._query_generation
                    )
                ):
                    break
                task_queue = self.task_results.get(source_id)
                if task_queue is not None:
                    # Queue.put is non-blocking.  Keeping generation validation
                    # and publication in the same critical section means close
                    # cannot race between them and accept an old collector's
                    # final result.
                    task_queue.put(task_copy)
                    continue

            LOGGER.warning(
                f'Ignore result for unknown source {source_id!r} in query generation '
                f'{query_generation!r}'
            )

    def fetch_visualization_data(self, source_id, task_queue=None):
        if task_queue is None:
            task_queue = self.task_results.get(source_id)
        if task_queue is None:
            return []
        tasks = task_queue.get_all()
        vis_results = []

        with Timer(f'Visualization preparation for {len(tasks)} tasks'):
            for idx, task in enumerate(tasks):
                file_path = self.get_file_result(task.get_file_path())
                try:
                    visualization_data = self.prepare_result_visualization_data(task, idx == len(tasks) - 1)
                except Exception as e:
                    LOGGER.warning(f'Prepare visualization data failed: {str(e)}')
                    LOGGER.exception(e)
                    continue

                FileOps.remove_file(file_path)

                vis_results.append({
                    'task_id': task.get_task_id(),
                    'data': visualization_data,
                })

        return vis_results

    def open_query(self, source_label):
        """Atomically open one datasource/result-collector generation."""
        source_config = self.find_datasource_configuration_by_label(source_label)
        if not source_config:
            return False, 'Datasource configuration not exists'

        with self.query_lock:
            if not self._query_admission_enabled or not self.result_url:
                return False, 'Runtime is not ready for datasource queries'
            if self.source_open:
                if self.source_label == source_label:
                    return True, 'Datasource is already open'
                return False, 'Another datasource is already open, please close it first'

            self._query_generation += 1
            generation = self._query_generation
            cancel_event = threading.Event()
            self._query_cancel_event = cancel_event
            self.source_open = True
            self.source_label = source_label
            source_ids = [source['id'] for source in source_config.get('source_list') or ()]
            self.task_results = {
                source_id: Queue(self._RESULT_WINDOW_SIZE)
                for source_id in source_ids
            }
            # Runtime endpoints are immutable for this install.  Capture the
            # distributor URL per generation so a cancelled collector cannot
            # refresh or overwrite lifecycle-owned URL bindings after uninstall.
            result_url = self.result_url
            self.is_get_result = True
            thread = threading.Thread(
                target=self.run_get_result,
                args=(generation, cancel_event, result_url),
                name=f'dayu-result-collector-{generation}',
                daemon=True,
            )
            self._result_thread = thread
            try:
                thread.start()
            except Exception:
                cancel_event.set()
                self.source_open = False
                self.source_label = ''
                self.is_get_result = False
                self.task_results.clear()
                self._query_cancel_event = None
                self._result_thread = None
                raise

        return True, 'Datasource open successfully'

    def _close_query_locked(self):
        """Cancel and clear the current generation while ``query_lock`` is held."""
        if self._query_cancel_event is not None:
            self._query_cancel_event.set()
        self._query_generation += 1
        self._query_cancel_event = None
        self._result_thread = None
        self.source_open = False
        self.source_label = ''
        self.is_get_result = False
        self.task_results.clear()
        self.customized_source_result_visualization_configs.clear()

    def close_query(self):
        """Cancel startup/collection immediately and clear its local state."""
        with self.query_lock:
            if not self.source_open and not self.is_get_result and self._query_cancel_event is None:
                return True, 'Datasource is already closed'
            self._close_query_locked()
        return True, 'Datasource close successfully'

    def query_snapshot(self, include_queues=False):
        """Return one internally consistent, immutable query-state snapshot."""
        with self.query_lock:
            return {
                'open': self.source_open,
                'source_label': self.source_label,
                'generation': self._query_generation,
                'queues': dict(self.task_results) if include_queues else None,
            }

    def is_query_generation_active(self, generation):
        with self.query_lock:
            return self.source_open and generation == self._query_generation

    def run_get_result(self, query_generation, cancel_event, result_url):
        cancel_event = cancel_event or threading.Event()
        time_ticket = 0
        try:
            while not cancel_event.wait(1):
                with self.query_lock:
                    if (
                        not self.is_get_result
                        or query_generation != self._query_generation
                        or cancel_event is not self._query_cancel_event
                    ):
                        return
                try:
                    response = http_request(result_url,
                                            method=NetworkAPIMethod.DISTRIBUTOR_RESULT,
                                            timeout=self._RESULT_REQUEST_TIMEOUT_SECONDS,
                                            json={
                                                'time_ticket': time_ticket,
                                                'size': self._RESULT_WINDOW_SIZE,
                                            })

                    if cancel_event.is_set():
                        return
                    if not response:
                        LOGGER.debug('[NO RESULT] Request result url failed.')
                        continue

                    time_ticket = response["time_ticket"]
                    results = response['result']
                    LOGGER.debug(f'Fetch {len(results)} tasks from time ticket: {time_ticket}')
                    self.parse_task_result(results, query_generation=query_generation)

                except Exception as e:
                    LOGGER.warning(f'Unexpected error occurred in getting task result: {str(e)}')
                    LOGGER.exception(e)
        finally:
            with self.query_lock:
                if (
                    query_generation == self._query_generation
                    and cancel_event is self._query_cancel_event
                ):
                    self.is_get_result = False
                    self._result_thread = None

    def _start_runtime_reconcile_loop(self, install_id):
        """Start the single publication/retirement worker for one installation."""
        install_id = str(install_id or '').strip()
        if not install_id:
            raise ValueError('runtime reconcile loop requires an install_id')
        # ``close`` publishes _closed first and then takes this same lock to
        # invalidate the worker. Whichever side wins is therefore linearized:
        # either startup observes closure, or close stops the newly made worker.
        with self._runtime_reconcile_lock:
            if self._closed:
                raise RuntimeError('backend lifecycle is closed')
            if self._runtime_reconcile_stop_event is not None:
                self._runtime_reconcile_stop_event.set()
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self.run_runtime_reconcile,
                args=(stop_event, install_id),
                name=f'dayu-runtime-reconcile-{install_id}',
                daemon=True,
            )
            self._runtime_reconcile_stop_event = stop_event
            self._runtime_reconcile_thread = thread
            try:
                thread.start()
            except Exception:
                stop_event.set()
                self._runtime_reconcile_stop_event = None
                self._runtime_reconcile_thread = None
                raise

    def _ensure_runtime_reconcile_loop(self, install_id):
        """Keep the existing lifecycle worker, or restore it if it exited."""
        with self._runtime_reconcile_lock:
            stop_event = self._runtime_reconcile_stop_event
            running = stop_event is not None and not stop_event.is_set()
        if not running:
            self._start_runtime_reconcile_loop(install_id)

    def _stop_runtime_reconcile_loop(self):
        """Invalidate the runtime worker before lifecycle mutation."""
        with self._runtime_reconcile_lock:
            if self._runtime_reconcile_stop_event is not None:
                self._runtime_reconcile_stop_event.set()
            self._runtime_reconcile_stop_event = None
            self._runtime_reconcile_thread = None

    @staticmethod
    def _runtime_progress_key(session):
        """Return lifecycle structure only; timestamps/errors are not progress."""
        if session is None:
            return None

        def runtime_ids(units):
            return tuple(
                getattr(unit, 'runtime_id', repr(unit)) for unit in (units or ())
            )

        retirement = getattr(session, 'retirement', None)
        uninstall = getattr(session, 'uninstall', None)
        return (
            getattr(session, 'install_id', ''),
            getattr(session, 'operation_id', ''),
            getattr(session, 'phase', ''),
            int(getattr(session, 'active_directory_revision', 0) or 0),
            runtime_ids(getattr(session, 'active', ())),
            runtime_ids(getattr(session, 'pending', ())),
            runtime_ids(getattr(session, 'cleanup', ())),
            (
                getattr(retirement, 'revision', 0),
                runtime_ids(getattr(retirement, 'units', ())),
            ) if retirement is not None else None,
            (
                bool(getattr(uninstall, 'deletion_submitted', False)),
                tuple(
                    (
                        getattr(resource, 'kind', ''),
                        getattr(resource, 'name', ''),
                        getattr(resource, 'uid', ''),
                    )
                    for resource in getattr(uninstall, 'remaining', ())
                ),
            ) if uninstall is not None else None,
        )

    def run_runtime_reconcile(self, stop_event, install_id):
        interval = max(0.0, float(self.processor_redeployment_interval_s))
        if interval <= 0:
            LOGGER.info('[Redeployment] Automatic processor rollout is disabled.')
        next_rollout = time.monotonic() + interval if interval > 0 else None
        retry_delay = 1.0
        try:
            while not stop_event.wait(retry_delay):
                session = None
                before_key = None
                cycle_failed = False
                progressed = False
                uninstall_cycle = False
                try:
                    with self._runtime_reconcile_lock:
                        if (
                            self._runtime_reconcile_stop_event is not stop_event
                            or stop_event.is_set()
                        ):
                            return
                    session = self.runtime_orchestrator.current_session()
                    if (
                        session is None
                        or session.install_id != install_id
                    ):
                        LOGGER.debug(
                            '[Runtime Reconcile] Managed runtime session changed; stop worker.'
                        )
                        return
                    before_key = self._runtime_progress_key(session)
                    if stop_event.is_set():
                        return
                    uninstall_cycle = session.phase in {
                        'uninstalling', 'finalizing-uninstall',
                    }
                    if uninstall_cycle:
                        cleanup_progressed = self.runtime_orchestrator.uninstall(
                            install_id,
                        )
                        session = self.runtime_orchestrator.current_session()
                        if session is None:
                            return
                        progressed = bool(cleanup_progressed)
                    else:
                        if session.phase == 'active':
                            progressed = self._ensure_local_runtime_projection(
                                session,
                                cancel_event=stop_event,
                            ) or progressed
                        changed = self.runtime_orchestrator.reconcile_retirement(
                            cancel_event=stop_event,
                        )
                        session = self.runtime_orchestrator.current_session()
                        if session is None or session.install_id != install_id:
                            return
                        if session.phase == 'active':
                            progressed = self._ensure_local_runtime_projection(
                                session,
                                cancel_event=stop_event,
                            ) or progressed
                        if (
                                session.phase == 'active'
                                and next_rollout is not None
                                and time.monotonic() >= next_rollout):
                            policy = self.find_scheduler_policy_by_id(session.policy_id)
                            result, message = self.parse_and_redeploy_services(
                                policy,
                                cancel_event=stop_event,
                            )
                            next_rollout = time.monotonic() + interval
                            if stop_event.is_set():
                                return
                            if not result:
                                cycle_failed = True
                                LOGGER.warning(f'[Redeployment] {message}')
                            session = self.runtime_orchestrator.current_session()
                            if session is None or session.install_id != install_id:
                                return
                        progressed = bool(changed) or progressed

                    if not uninstall_cycle:
                        progressed = (
                            progressed
                            or self._runtime_progress_key(session) != before_key
                        )
                    if (
                            session is not None
                            and session.phase in {'uninstalling', 'finalizing-uninstall'}
                            and not progressed):
                        # Remaining cleanup is expected, not an exception. It
                        # still uses the existing reconcile backoff so a stuck
                        # resource cannot turn frontend polling into Kubernetes
                        # list traffic.
                        cycle_failed = True
                    deferred_failure = bool(
                        session is not None
                        and getattr(session, 'last_error', '')
                        and (
                            getattr(session, 'cleanup', ())
                            or getattr(session, 'retirement', None) is not None
                            or str(getattr(session, 'phase', '')).startswith('publishing')
                        )
                    )
                    if deferred_failure and not progressed:
                        cycle_failed = True
                except Exception as exc:
                    try:
                        current = self.runtime_orchestrator.current_session()
                    except Exception:
                        current = None
                    progressed = (
                        not uninstall_cycle
                        and before_key is not None
                        and self._runtime_progress_key(current) != before_key
                    )
                    cycle_failed = True
                    if interval > 0:
                        next_rollout = time.monotonic() + interval
                    LOGGER.warning(f'[Runtime Reconcile] Unexpected error: {exc}')
                    LOGGER.exception(exc)
                if cycle_failed and not progressed:
                    retry_delay = min(retry_delay * 2.0, 30.0)
                else:
                    retry_delay = 1.0
        finally:
            with self._runtime_reconcile_lock:
                if self._runtime_reconcile_stop_event is stop_event:
                    self._runtime_reconcile_stop_event = None
                    self._runtime_reconcile_thread = None

    @staticmethod
    def _count_jsonl_records(file_path):
        if not os.path.exists(file_path):
            return 0
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    def _append_system_log_snapshot(self, snapshot):
        with open(self.system_log_store_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snapshot, ensure_ascii=False))
            f.write('\n')

    def _maybe_compact_system_log_store_locked(self):
        if not self.system_log_retention_records:
            return
        if self.system_log_record_count <= self.system_log_retention_records + self.system_log_compact_interval:
            return

        recent_lines = deque(maxlen=self.system_log_retention_records)
        try:
            with open(self.system_log_store_path, 'r', encoding='utf-8') as src:
                for line in src:
                    line = line.strip()
                    if line:
                        recent_lines.append(line)

            temp_handle = tempfile.NamedTemporaryFile(
                prefix='dayu-system-log-compact-',
                suffix='.jsonl',
                delete=False
            )
            temp_path = temp_handle.name
            temp_handle.close()

            try:
                with open(temp_path, 'w', encoding='utf-8') as dst:
                    for line in recent_lines:
                        dst.write(line)
                        dst.write('\n')
                os.replace(temp_path, self.system_log_store_path)
            except Exception:
                FileOps.remove_file(temp_path)
                raise

            self.system_log_record_count = len(recent_lines)
            LOGGER.info(f'[Backend] Compacted system log store to {self.system_log_record_count} records.')
        except Exception as e:
            LOGGER.warning(f'Compact system log store failed: {str(e)}')
            LOGGER.exception(e)

    def _create_system_log_snapshot_file(self):
        snapshot_handle = tempfile.NamedTemporaryFile(
            prefix='dayu-system-log-snapshot-',
            suffix='.jsonl',
            delete=False
        )
        snapshot_path = snapshot_handle.name
        snapshot_handle.close()

        with self.system_log_lock:
            if os.path.exists(self.system_log_store_path):
                shutil.copyfile(self.system_log_store_path, snapshot_path)
            else:
                with open(snapshot_path, 'w', encoding='utf-8'):
                    pass

        return snapshot_path

    def create_system_log_export_file(self):
        snapshot_path = self._create_system_log_snapshot_file()
        export_handle = tempfile.NamedTemporaryFile(
            prefix='dayu-system-log-',
            suffix='.json.gz',
            delete=False
        )
        export_path = export_handle.name
        export_handle.close()

        try:
            with gzip.open(export_path, 'wt', encoding='utf-8') as fh:
                fh.write('[\n')
                first = True
                with open(snapshot_path, 'r', encoding='utf-8') as src:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            LOGGER.warning('[Backend] Skip malformed system log record during export.')
                            continue

                        if not first:
                            fh.write(',\n')
                        fh.write(_indent_json_block(json.dumps(record, ensure_ascii=False, indent=4)))
                        first = False

                if not first:
                    fh.write('\n')
                fh.write(']\n')
        except Exception:
            FileOps.remove_file(export_path)
            raise
        finally:
            FileOps.remove_file(snapshot_path)

        return export_path

    def get_system_parameters(self):
        # A durable active Session is insufficient: only the exact directory
        # generation projected by this Backend process owns local telemetry.
        _, _, local_ready, _ = self.management_lifecycle_snapshot()
        if not local_ready:
            return []

        # Backend-controlled timestamp; scheduler values are already cached.
        timestamp = time.strftime('%H:%M:%S', time.localtime())

        data = self.prepare_system_visualizations_data()
        snapshot = {"timestamp": timestamp, "data": data}

        try:
            with self.system_log_lock:
                self._append_system_log_snapshot(snapshot)
                self.system_log_record_count += 1
                self._maybe_compact_system_log_store_locked()
        except Exception as e:
            LOGGER.warning(f'Append system log failed: {str(e)}')
            LOGGER.exception(e)

        return [snapshot]

    def get_runtime_telemetry(self, logical_service=''):
        return self.runtime_telemetry.snapshot(logical_service=logical_service)

    def close(self):
        with self._lifecycle_control_lock:
            self._closed = True
            self._runtime_recovery_stop_event.set()
            self._runtime_recovery_wake_event.set()
            if self._install_admission is not None:
                self._install_admission.cancel()
            self._bound_runtime_key = None
        self._stop_runtime_reconcile_loop()
        with self.query_lock:
            self._query_admission_enabled = False
            self._close_query_locked()
        self.runtime_telemetry.unbind()
        self.runtime_telemetry.close()

    def check_datasource_config(self, config_path):
        if not YamlOps.is_yaml_file(config_path):
            return None

        config = YamlOps.read_yaml(config_path)
        try:
            _ = config['source_name']
            _ = config['source_type']
            _ = config['source_mode']
            for camera in config['source_list']:
                _ = camera['name']
                if self.inner_datasource:
                    _ = camera['dir']
                else:
                    _ = camera['url']
                _ = camera['metadata']

        except Exception as e:
            LOGGER.warning(f'Datasource config file format error: {str(e)}')
            LOGGER.exception(e)
            return None

        return config

    def check_visualization_config(self, config_path):
        if not YamlOps.is_yaml_file(config_path):
            return None

        config = YamlOps.read_yaml(config_path)

        try:
            for visualization in config:
                viz_name = visualization['name']
                assert isinstance(viz_name, str), '"name" is not a string'
                viz_type = visualization['type']
                assert isinstance(viz_type, str), '"type" is not a string'
                viz_var = visualization['variables']
                assert isinstance(viz_var, list), '"variables" is not a list'
                viz_size = visualization['size']
                assert isinstance(viz_size, int), '"size" is not an integer'
                if 'hook_name' in visualization:
                    assert isinstance(visualization['hook_name'], str), '"hook_name" is not a string'
                if 'hook_params' in visualization:
                    assert isinstance(visualization['hook_params'], str), '"hook_params" is not a string(dict)'
                    assert isinstance(eval(visualization['hook_params']), dict), '"hook_params" is not a string(dict)'
                if 'x_axis' in visualization:
                    assert isinstance(visualization['x_axis'], str), '"x_axis" is not a string'
                if 'y_axis' in visualization:
                    assert isinstance(visualization['y_axis'], str), '"y_axis" is not a string'
            return config
        except Exception as e:
            LOGGER.warning(f'Visualization config file format error: {str(e)}')
            LOGGER.exception(e)
            return None

    @staticmethod
    def _runtime_unit(directory, component):
        matches = [unit for unit in directory.routes if unit.slot.component == component]
        if len(matches) != 1 or matches[0].endpoint is None:
            raise RuntimeError(f'RuntimeDirectory requires exactly one endpoint for {component!r}')
        return matches[0]

    def _bind_runtime_urls(self, directory):
        scheduler = self._runtime_unit(directory, 'scheduler').endpoint
        distributor = self._runtime_unit(directory, 'distributor').endpoint
        scheduler_base = f'http://{scheduler.url_authority}'
        distributor_base = f'http://{distributor.url_authority}'
        self.resource_url = f'{scheduler_base}{NetworkAPIPath.SCHEDULER_GET_RESOURCE}'
        self.runtime_telemetry.bind(self.resource_url, directory)
        self.result_url = f'{distributor_base}{NetworkAPIPath.DISTRIBUTOR_RESULT}'
        self.result_file_url = f'{distributor_base}{NetworkAPIPath.DISTRIBUTOR_FILE}'
        self.log_fetch_url = f'{distributor_base}{NetworkAPIPath.DISTRIBUTOR_EXPORT_RESULT_LOG}'

    def get_file_result(self, file_name):
        if not self.result_file_url:
            return ''
        response = http_request(self.result_file_url,
                                method=NetworkAPIMethod.DISTRIBUTOR_FILE,
                                no_decode=True,
                                json={'file': file_name},
                                stream=True)
        if response is None:
            self.result_file_url = None
            return ''
        with open(file_name, 'wb') as file_out:
            for chunk in response.iter_content(chunk_size=8192):
                file_out.write(chunk)
        return file_name

    def open_result_log_export_stream(self):
        self.parse_base_info()
        _, _, local_ready, _ = self.management_lifecycle_snapshot()
        if not local_ready or not self.log_fetch_url:
            return None

        response = http_request(
            self.log_fetch_url,
            method=NetworkAPIMethod.DISTRIBUTOR_EXPORT_RESULT_LOG,
            no_decode=True,
            stream=True
        )
        if response is None:
            self.log_fetch_url = None
            return None
        return response

    def get_result_visualization_config(self, source_id):
        self.parse_base_info()
        visualizations = self.customized_source_result_visualization_configs[source_id] \
            if source_id in self.customized_source_result_visualization_configs else self.result_visualization_configs
        return [{'id': idx, **vf} for idx, vf in enumerate(visualizations)]

    def get_system_visualization_config(self):
        self.parse_base_info()
        return [{'id': idx, **vf} for idx, vf in enumerate(self.system_visualization_configs)]
