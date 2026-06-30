import copy
import logging


LOGGER = logging.getLogger(__name__)


class MadEyePolicySearch:
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
                 madeye_param=None,
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
        madeye_param = copy.deepcopy(madeye_param) if madeye_param else {}
        self.max_iterations = int(search_param.get('max_iterations', 32))
        self.max_expanded_states = int(search_param.get('max_expanded_states', 512))
        self.min_improvement = float(search_param.get('min_improvement', 1e-9))
        self.candidate_pool_size = int(madeye_param.get(
            'candidate_pool_size',
            search_param.get('candidate_pool_size', 8)
        ))

        if performance_predictor is None:
            from ..steady import IntegratedSafePredictor

            performance_predictor = IntegratedSafePredictor(
                kb_path=kb_path,
                service_name_pipeline=service_name_pipeline,
                corrector_param=corrector_param,
                queue_param=queue_param,
            )
        self.performance_predictor = performance_predictor
        self.candidate_pool = {}

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
            LOGGER.warning(f'[MadEye] Failed to update predictor feedback: {exc}')

    def get_schedule_plan(self, cur_task_id, cur_policy, context_info):
        if cur_task_id is not None:
            self.latest_task_id = int(cur_task_id)
        if cur_policy is None or context_info is None:
            return copy.deepcopy(self.default_policy)

        cur_policy = self._clean_policy(cur_policy)
        pool = {}
        for candidate in list(self.candidate_pool.values()):
            self._add_candidate(pool, candidate['policy'], self.evaluate_loss(candidate['policy'], context_info))
        self._add_candidate(pool, cur_policy, self.evaluate_loss(cur_policy, context_info))
        pool = self._trim_pool(pool)

        best = self._best_candidate(pool)
        if best is None:
            return copy.deepcopy(self.default_policy)

        expanded_states = 0
        for _ in range(self.max_iterations):
            next_pool = copy.deepcopy(pool)
            for candidate in self._sorted_candidates(pool):
                for neighbor in self.get_neighbors(candidate['policy']):
                    self._add_candidate(next_pool, neighbor, self.evaluate_loss(neighbor, context_info))
                    expanded_states += 1
                    if expanded_states >= self.max_expanded_states:
                        break
                if expanded_states >= self.max_expanded_states:
                    break

            next_pool = self._trim_pool(next_pool)
            next_best = self._best_candidate(next_pool)
            self.candidate_pool = copy.deepcopy(next_pool)

            if next_best is None or not self._is_improved(next_best['loss'], best['loss']):
                break

            pool = next_pool
            best = next_best
            if expanded_states >= self.max_expanded_states:
                break

        return copy.deepcopy(best['policy'])

    def get_neighbors(self, policy):
        neighbors = []
        for knob in self.knob_names:
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
            LOGGER.debug(f'[MadEye] Failed to evaluate policy {policy}: {exc}')
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

    def _add_candidate(self, pool, policy, loss):
        policy = self._clean_policy(policy)
        key = self._policy_key(policy)
        candidate = {'policy': policy, 'loss': float(loss)}
        if key not in pool or candidate['loss'] < pool[key]['loss']:
            pool[key] = candidate

    def _trim_pool(self, pool):
        return {
            self._policy_key(candidate['policy']): candidate
            for candidate in self._sorted_candidates(pool)[:max(1, self.candidate_pool_size)]
        }

    @staticmethod
    def _sorted_candidates(pool):
        return sorted(copy.deepcopy(list(pool.values())), key=lambda item: item['loss'])

    @staticmethod
    def _best_candidate(pool):
        candidates = MadEyePolicySearch._sorted_candidates(pool)
        return candidates[0] if candidates else None

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
