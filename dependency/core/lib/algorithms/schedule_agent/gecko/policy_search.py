import copy
import logging
import math


LOGGER = logging.getLogger(__name__)


class GeckoPolicySearch:
    def __init__(self,
                 kb_path,
                 service_name_pipeline,
                 knob_value_range_dict,
                 delay_cons,
                 acc_cons,
                 delay_weight,
                 acc_weight,
                 default_policy,
                 raw_meta_data,
                 corrector_param,
                 queue_param,
                 search_param=None,
                 gecko_param=None,
                 use_corrected_prediction=False,
                 enable_feedback_update=False,
                 performance_predictor=None):
        self.service_name_pipeline = self._trim_pipeline(service_name_pipeline)
        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)
        self.knob_names = list(self.knob_value_range_dict.keys())
        self.delay_cons = delay_cons
        self.acc_cons = acc_cons
        self.delay_weight = delay_weight
        self.acc_weight = acc_weight
        self.default_policy = self._clean_policy(default_policy)
        self.raw_meta_data = copy.deepcopy(raw_meta_data) if raw_meta_data else {}
        self.use_corrected_prediction = bool(use_corrected_prediction)
        self.enable_feedback_update = bool(enable_feedback_update)
        self.latest_task_id = -1

        search_param = copy.deepcopy(search_param) if search_param else {}
        gecko_param = copy.deepcopy(gecko_param) if gecko_param else {}
        self.max_iterations = int(gecko_param.get('max_iterations',
                                                  search_param.get('max_iterations', 32)))
        self.max_expanded_states = int(gecko_param.get('max_expanded_states',
                                                       search_param.get('max_expanded_states', 512)))
        self.min_improvement = float(gecko_param.get('min_improvement',
                                                     search_param.get('min_improvement', 1e-9)))
        self.speed_threshold = float(gecko_param.get('speed_threshold',
                                                     search_param.get('speed_threshold', 1.0)))
        self.slowdown_factor = float(gecko_param.get('slowdown_factor',
                                                     search_param.get('slowdown_factor', 0.5)))

        if performance_predictor is None:
            from ..steady import IntegratedSafePredictor

            performance_predictor = IntegratedSafePredictor(
                kb_path=kb_path,
                service_name_pipeline=service_name_pipeline,
                corrector_param=corrector_param,
                queue_param=queue_param,
            )
        self.performance_predictor = performance_predictor

    def update_feedback(self, context_info, conf_info, task_info):
        if not self.enable_feedback_update:
            return
        if context_info is None or conf_info is None or task_info is None:
            return

        self.latest_task_id = int(task_info.get('task_id', self.latest_task_id))
        try:
            self.performance_predictor.update_corrector(
                context_info=context_info,
                conf_info=conf_info,
                task_info=task_info,
            )
        except Exception as exc:
            LOGGER.warning(f'[Gecko] Failed to update predictor feedback: {exc}')

    def get_schedule_plan(self, cur_task_id, cur_policy, context_info):
        if cur_task_id is not None:
            self.latest_task_id = int(cur_task_id)
        if cur_policy is None or context_info is None:
            return copy.deepcopy(self.default_policy)

        adjusted_policy = self._clean_policy(cur_policy)
        adjusted_policy = self._adjust_fps(adjusted_policy, context_info)
        adjusted_policy = self._adjust_resolution(adjusted_policy, context_info)

        searchable_knobs = [
            knob for knob in self.knob_names
            if knob not in ('fps', 'resolution')
        ]
        return self._hill_climb(adjusted_policy, context_info, searchable_knobs)

    def _hill_climb(self, start_policy, context_info, knob_names):
        best_policy = copy.deepcopy(start_policy)
        best_loss = self.evaluate_loss(best_policy, context_info)
        visited = {self._policy_key(best_policy)}
        expanded_states = 0

        for _ in range(self.max_iterations):
            round_best_policy = best_policy
            round_best_loss = best_loss

            for neighbor in self.get_neighbors(best_policy, knob_names):
                neighbor_key = self._policy_key(neighbor)
                if neighbor_key in visited:
                    continue
                visited.add(neighbor_key)

                loss = self.evaluate_loss(neighbor, context_info)
                expanded_states += 1
                if self._is_improved(loss, round_best_loss):
                    round_best_policy = neighbor
                    round_best_loss = loss
                if expanded_states >= self.max_expanded_states:
                    break

            if not self._is_improved(round_best_loss, best_loss):
                break

            best_policy = copy.deepcopy(round_best_policy)
            best_loss = round_best_loss
            if expanded_states >= self.max_expanded_states:
                break

        return copy.deepcopy(best_policy)

    def _adjust_fps(self, policy, context_info):
        if 'fps' not in policy or 'fps' not in self.knob_value_range_dict:
            return policy

        values = self.knob_value_range_dict['fps']
        try:
            cur_index = values.index(policy['fps'])
        except ValueError:
            return policy

        obj_speed = self._as_float(context_info.get('obj_speed'), default=0.0)
        if obj_speed <= self.speed_threshold:
            new_index = int(math.floor(cur_index * self.slowdown_factor))
        else:
            new_index = cur_index + 1

        new_index = min(len(values) - 1, max(0, new_index))
        policy['fps'] = values[new_index]
        return policy

    def _adjust_resolution(self, policy, context_info):
        if 'resolution' not in policy or 'resolution' not in self.knob_value_range_dict:
            return policy

        values = self.knob_value_range_dict['resolution']
        try:
            cur_index = values.index(policy['resolution'])
        except ValueError:
            return policy

        obj_num = self._as_float(context_info.get('obj_num'), default=0.0)
        if obj_num <= 0:
            new_index = int(math.floor(cur_index * self.slowdown_factor))
        else:
            new_index = cur_index + 1

        new_index = min(len(values) - 1, max(0, new_index))
        policy['resolution'] = values[new_index]
        return policy

    def get_neighbors(self, policy, knob_names):
        neighbors = []
        for knob in knob_names:
            if knob not in policy:
                continue

            values = self.knob_value_range_dict[knob]
            try:
                cur_index = values.index(policy[knob])
            except ValueError:
                continue

            for direction in (-1, 1):
                new_index = cur_index + direction
                if 0 <= new_index < len(values):
                    neighbor = copy.deepcopy(policy)
                    neighbor[knob] = values[new_index]
                    neighbors.append(neighbor)
        return neighbors

    def evaluate_loss(self, policy, context_info):
        try:
            delay, acc = self.predict_delay_accuracy(policy, context_info)
        except Exception as exc:
            LOGGER.debug(f'[Gecko] Failed to evaluate policy {policy}: {exc}')
            return float('inf')

        delay_loss = 0.0
        if self.delay_cons > 0 and delay > self.delay_cons:
            delay_loss = (delay - self.delay_cons) / self.delay_cons

        acc_loss = 0.0
        if self.acc_cons > 0 and acc < self.acc_cons:
            acc_loss = (self.acc_cons - acc) / self.acc_cons

        return delay_loss * self.delay_weight + acc_loss * self.acc_weight

    def predict_delay_accuracy(self, policy, context_info):
        delay = self.performance_predictor.delay_pre(
            context_info=context_info,
            conf_info=policy,
            latest_task_id=self.latest_task_id + 1,
            if_correct=self.use_corrected_prediction,
        )
        raw_fps = self.raw_meta_data.get('fps') or policy.get('fps') or 1
        delay *= policy.get('fps', raw_fps) / raw_fps
        acc = self.performance_predictor.acc_pre(
            context_info=context_info,
            conf_info=policy,
            if_correct=self.use_corrected_prediction,
        )
        return max(0.0, delay), acc

    def _clean_policy(self, policy):
        return {
            knob: copy.deepcopy(policy[knob])
            for knob in self.knob_names
            if knob in policy
        }

    def _policy_key(self, policy):
        return tuple((knob, repr(policy.get(knob))) for knob in sorted(self.knob_names))

    def _is_improved(self, new_loss, old_loss):
        return new_loss + self.min_improvement < old_loss

    @staticmethod
    def _trim_pipeline(service_name_pipeline):
        trimmed = []
        for service_name in service_name_pipeline:
            if service_name == 'end':
                break
            trimmed.append(service_name)
        return trimmed

    @staticmethod
    def _as_float(value, default=0.0):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = sum(value) / len(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
