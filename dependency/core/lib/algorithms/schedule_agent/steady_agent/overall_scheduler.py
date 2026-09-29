import copy
import math
import threading
import time

from core.lib.common import LOGGER

from .knowledge_base import KnowledgeBase
from .steady_record import SteadyRecord


class MacroSearch:
    def __init__(
        self,
        kb_path,
        service_name_pipeline,
        corrector_param,
        queue_param,
        knob_value_range_dict,
        delay_cons,
        acc_cons,
        delay_weight,
        acc_weight,
        default_policy,
        raw_meta_data,
        context_names,
        history_length,
        stop_threshold,
        context_analyze_type,
        cluster_threshold,
        if_online_train,
    ):

        self.knowledge_base = KnowledgeBase(
            kb_path=kb_path,
            service_name_pipeline=service_name_pipeline,
            corrector_param=corrector_param,
            queue_param=queue_param,
            knob_value_range_dict=knob_value_range_dict,
            delay_cons=delay_cons,
            acc_cons=acc_cons,
            delay_weight=delay_weight,
            acc_weight=acc_weight,
            raw_meta_data=raw_meta_data,
            stop_threshold=stop_threshold,
            cluster_threshold=cluster_threshold,
            if_online_train=if_online_train,
        )

        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)

        self.conv_policy = copy.deepcopy(default_policy)

        self.context_history = {}
        self.history_length = history_length
        for context in context_names:
            self.context_history[context] = []

        self.context_analyze_type = context_analyze_type

    def update_context(self, context_info):

        self.knowledge_base.update_context_for_classifier_train(context_info=context_info)

        if context_info is not None:
            for context in self.context_history.keys():
                self.context_history[context].append(context_info[context])

                if len(self.context_history[context]) > self.history_length:
                    self.context_history[context].pop(0)
        else:
            for context in self.context_history.keys():
                while len(self.context_history[context]) > self.history_length:
                    self.context_history[context].pop(0)

    def get_analyzed_context(self):

        analyzed_context = {}

        if self.context_analyze_type == 1:
            for context in self.context_history.keys():
                if len(self.context_history[context]) == 0:
                    analyzed_context = {}
                    break

                else:
                    analyzed_context[context] = sum(self.context_history[context]) / len(self.context_history[context])

        else:
            for context in self.context_history.keys():
                if len(self.context_history[context]) == 0:
                    analyzed_context = {}
                    break

                else:
                    if context in ["band_Mbps", "obj_size_norm"]:
                        analyzed_context[context] = min(self.context_history[context])

                    elif context in ["obj_num", "obj_speed"]:
                        analyzed_context[context] = max(self.context_history[context])

        return analyzed_context

    def update_conv_policy(self, start_policy, real_time_delay, real_time_acc):

        cur_context = self.get_analyzed_context()

        path_record = []

        if cur_context is not None:
            if len(cur_context) > 0:
                delay, acc = self.pre_delay_and_acc(context_info=cur_context, conf_info=start_policy)

                start_loss = self.cal_loss(delay=delay, acc=acc)

                sorted_knob_list = self.knowledge_base.get_sorted_knob_list(
                    cur_policy=start_policy,
                    cur_context=cur_context,
                    real_time_delay=real_time_delay,
                    real_time_acc=real_time_acc,
                )
                if_greedy_for_path_record = 1

                if sorted_knob_list is not None:
                    if len(sorted_knob_list) > 0:
                        path_record = self.knowledge_base.sorted_search(
                            policy=start_policy,
                            cur_context=cur_context,
                            loss=start_loss,
                            sorted_knob_list=sorted_knob_list,
                        )
                        if_greedy_for_path_record = 0

                if if_greedy_for_path_record == 1:
                    all_knob_list = list(self.knob_value_range_dict.keys())
                    path_record = self.knowledge_base.greedy_search(
                        policy=start_policy, cur_context=cur_context, loss=start_loss, all_knob_list=all_knob_list
                    )

                if len(path_record) > 0:
                    self.conv_policy = copy.deepcopy(path_record[-1]["policy"])
                    copy.deepcopy(path_record[-1]["loss"])

        return path_record

    def choose_knobs_by_conv_policy(self, cur_policy):
        chosen_knob_list = []
        for knob in list(cur_policy.keys()):
            if self.conv_policy[knob] != cur_policy[knob]:
                chosen_knob_list.append(knob)

        return chosen_knob_list

    def update_corrector(self, context_info, conf_info, task_info):

        self.knowledge_base.update_corrector(context_info=context_info, conf_info=conf_info, task_info=task_info)

    def get_delay_decrease_and_acc_increase_weight(self, cur_policy, cur_context, chosen_knob_list):
        norm_delay_decrease_weight, norm_acc_increase_weight = (
            self.knowledge_base.get_delay_decrease_and_acc_increase_weight(
                cur_policy=cur_policy, cur_context=cur_context, chosen_knob_list=chosen_knob_list
            )
        )
        return norm_delay_decrease_weight, norm_acc_increase_weight

    def pre_delay_and_acc(self, context_info, conf_info, record_path=None):
        delay, acc = self.knowledge_base.pre_delay_and_acc(
            context_info=context_info, conf_info=conf_info, record_path=record_path
        )
        return delay, acc

    def cal_loss(self, delay, acc):
        loss = self.knowledge_base.cal_loss(delay=delay, acc=acc)
        return loss


class MicroFeedback:
    def __init__(self, coeff_info, knob_value_range_dict):

        self.coeff_info = copy.deepcopy(coeff_info)
        self.coeff_info["step_coeff"]["cur_value"] = self.coeff_info["step_coeff"]["start_value"]

        self.delay_loss = 0
        self.acc_loss = 0

        self.all_knob_weight = {}

        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)

    def update_delay_loss_and_acc_loss(self, delay_loss, acc_loss):
        self.delay_loss = delay_loss
        self.acc_loss = acc_loss

    def update_step_coeff(self, if_meet_cons_before, if_meet_cons_now):

        if (not if_meet_cons_before) and (not if_meet_cons_now):
            self.update_coeff(coeff_name="step_coeff", if_add=True)

        elif (not if_meet_cons_before) and if_meet_cons_now:
            self.update_coeff(coeff_name="step_coeff", if_add=False)

        elif if_meet_cons_before and (not if_meet_cons_now):
            self.update_coeff(coeff_name="step_coeff", if_add=True)

        elif if_meet_cons_before and if_meet_cons_now:
            pass

    def update_all_knob_weight(self, delay_decrease_weight, acc_increase_weight):

        if self.delay_loss + self.acc_loss == 0:
            delay_perf = 0
            acc_perf = 0
        else:
            delay_perf = self.delay_loss / (self.delay_loss + self.acc_loss)
            acc_perf = self.acc_loss / (self.delay_loss + self.acc_loss)

        for knob in delay_decrease_weight.keys():
            self.all_knob_weight[knob] = acc_perf * acc_increase_weight[knob] + delay_perf * delay_decrease_weight[knob]

    def update_coeff(self, coeff_name, if_add):

        if if_add:
            if (
                self.coeff_info[coeff_name]["cur_value"] + self.coeff_info[coeff_name]["add_interval"]
                <= self.coeff_info[coeff_name]["max_value"]
            ):
                self.coeff_info[coeff_name]["cur_value"] += self.coeff_info[coeff_name]["add_interval"]
            else:
                self.coeff_info[coeff_name]["cur_value"] = self.coeff_info[coeff_name]["max_value"]
        else:
            if (self.coeff_info[coeff_name]["cur_value"] / 2.0) >= self.coeff_info[coeff_name]["min_value"]:
                self.coeff_info[coeff_name]["cur_value"] /= 2.0
            else:
                self.coeff_info[coeff_name]["cur_value"] = self.coeff_info[coeff_name]["min_value"]

    def get_new_policy_by_feedback(self, old_policy):

        new_policy = copy.deepcopy(old_policy)

        for knob in list(self.knob_value_range_dict.keys()):
            if knob == "encoding":
                continue

            knob_value_range = list(self.knob_value_range_dict[knob])

            old_knob_value = old_policy[knob]
            old_knob_idx = knob_value_range.index(old_knob_value)

            step_coeff = self.coeff_info["step_coeff"]["cur_value"]

            knob_weight = self.all_knob_weight[knob]

            new_knob_idx = min(
                max(int(old_knob_idx + (self.delay_loss + self.acc_loss) * step_coeff * knob_weight), 0),
                len(knob_value_range) - 1,
            )

            new_knob_value = knob_value_range[new_knob_idx]

            new_policy[knob] = new_knob_value

        return new_policy


class OverallScheduler:
    """Coordinate macro search, micro feedback and the two UNSTEADY ablations."""

    def __init__(
        self,
        kb_path,
        service_name_pipeline,
        corrector_param,
        queue_param,
        knob_value_range_dict,
        delay_cons,
        acc_cons,
        delay_weight,
        acc_weight,
        default_policy,
        raw_meta_data,
        context_names,
        history_length,
        stop_threshold,
        macro_update_interval,
        context_analyze_type,
        coeff_info,
        steady_record_path,
        correct_record_path,
        cluster_threshold,
        if_online_train,
        schedule_type="steady",
        feedback_weight=None,
    ):
        if schedule_type not in ("steady", "macro", "micro"):
            raise ValueError("schedule_type must be steady, macro or micro")
        self.schedule_type = schedule_type
        self.feedback_weight = copy.deepcopy(feedback_weight)
        if schedule_type == "micro":
            for name in ("delay_decrease_weight", "acc_increase_weight"):
                weights = (self.feedback_weight or {}).get(name, {})
                if set(weights) != set(knob_value_range_dict) or not all(
                    math.isfinite(value) for value in weights.values()
                ):
                    raise ValueError(f"feedback_weight.{name} must cover every scheduling knob with finite weights")
        self.default_policy = copy.deepcopy(default_policy)
        self.stop_threshold = stop_threshold
        self.steady_record_path = steady_record_path
        self.correct_record_path = correct_record_path
        self.latest_real_time_loss = None
        self.macro_search = MacroSearch(
            kb_path=kb_path,
            service_name_pipeline=service_name_pipeline,
            corrector_param=corrector_param,
            queue_param=queue_param,
            knob_value_range_dict=knob_value_range_dict,
            delay_cons=delay_cons,
            acc_cons=acc_cons,
            delay_weight=delay_weight,
            acc_weight=acc_weight,
            default_policy=default_policy,
            raw_meta_data=raw_meta_data,
            context_names=context_names,
            history_length=history_length,
            stop_threshold=stop_threshold,
            context_analyze_type=context_analyze_type,
            cluster_threshold=cluster_threshold,
            if_online_train=if_online_train,
        )
        self.micro_feedback = MicroFeedback(coeff_info, knob_value_range_dict)
        self._lock = threading.Lock()
        self._search_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._parameters = None
        self._macro_output = None
        self._generation = 0
        self.macro_update_interval = macro_update_interval
        self._thread = None
        if schedule_type != "micro":
            self._thread = threading.Thread(target=self._compute_loop, daemon=True)
            self._thread.start()

    def update_constraints(self, delay_cons, acc_cons, delay_weight, acc_weight):
        with self._search_lock:
            knowledge = self.macro_search.knowledge_base
            knowledge.update_delay_cons(delay_cons)
            knowledge.update_acc_cons(acc_cons)
            knowledge.update_delay_weight(delay_weight)
            knowledge.update_acc_weight(acc_weight)
            with self._lock:
                self._generation += 1
                self._macro_output = None
                self.latest_real_time_loss = None

    def update_parameter(self, cur_policy, real_time_context_info, real_time_delay, real_time_acc):
        with self._lock:
            self._parameters = copy.deepcopy((cur_policy, real_time_context_info, real_time_delay, real_time_acc))

    def get_macro_output(self):
        with self._lock:
            return copy.deepcopy(self._macro_output)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.macro_search.knowledge_base.stop()

    def _compute_macro(self, policy, context, delay, accuracy):
        search = self.macro_search
        analyzed = search.get_analyzed_context()
        previous = self.get_macro_output()
        needs_search = previous is None or not analyzed
        if not needs_search:
            predicted_delay, predicted_accuracy = search.pre_delay_and_acc(analyzed, search.conv_policy)
            converged_loss = search.cal_loss(predicted_delay, predicted_accuracy)
            predicted_delay, predicted_accuracy = search.pre_delay_and_acc(analyzed, policy)
            current_loss = search.cal_loss(predicted_delay, predicted_accuracy)
            needs_search = converged_loss > current_loss * (1 - self.stop_threshold)
        path = search.update_conv_policy(search.conv_policy, delay, accuracy) if needs_search else []
        knobs = search.choose_knobs_by_conv_policy(policy)
        delay_weights, accuracy_weights = search.get_delay_decrease_and_acc_increase_weight(policy, context, knobs)
        return {
            "if_need_macro_search": needs_search,
            "path_record": path,
            "context_history": copy.deepcopy(search.context_history),
            "chosen_knob_list": knobs,
            "delay_decrease_weight": delay_weights,
            "acc_increase_weight": accuracy_weights,
        }

    def _compute_loop(self):
        while not self._stop_event.wait(self.macro_update_interval):
            with self._lock:
                parameters = copy.deepcopy(self._parameters)
                generation = self._generation
            if parameters is None or any(value is None for value in parameters):
                continue
            started = time.monotonic()
            try:
                with self._search_lock:
                    output = self._compute_macro(*parameters)
                output["macro_search_delay"] = time.monotonic() - started
                with self._lock:
                    if generation == self._generation:
                        self._macro_output = output
            except Exception:
                LOGGER.exception("STEADY macro search failed; retaining the last scheduling decision")

    def update_scheduler(self, context_info, conf_info, task_info):
        if context_info is None or conf_info is None or task_info is None:
            return
        with self._search_lock:
            self.macro_search.update_context(context_info)
            self.macro_search.update_corrector(context_info, conf_info, task_info)

    def _feedback_policy(self, policy, delay, accuracy, weights):
        knowledge = self.macro_search.knowledge_base
        loss = knowledge.cal_loss(delay, accuracy)
        self.micro_feedback.update_delay_loss_and_acc_loss(
            knowledge.cal_delay_loss(delay), knowledge.cal_acc_loss(accuracy)
        )
        self.micro_feedback.update_step_coeff(self.latest_real_time_loss == 0, loss == 0)
        self.latest_real_time_loss = loss
        self.micro_feedback.update_all_knob_weight(weights["delay_decrease_weight"], weights["acc_increase_weight"])
        return self.micro_feedback.get_new_policy_by_feedback(policy), loss

    def get_schedule_plan(self, cur_task_id, cur_policy, context_info, real_time_delay, real_time_acc):
        self.update_parameter(cur_policy, context_info, real_time_delay, real_time_acc)
        if any(value is None for value in (cur_policy, context_info, real_time_delay, real_time_acc)):
            return copy.deepcopy(self.default_policy)
        if self.schedule_type == "micro":
            policy, _ = self._feedback_policy(cur_policy, real_time_delay, real_time_acc, self.feedback_weight)
            return policy
        macro = self.get_macro_output()
        if macro is None:
            return copy.deepcopy(cur_policy)
        if self.schedule_type == "macro":
            path = macro["path_record"]
            return copy.deepcopy(path[-1]["policy"] if path else cur_policy)
        policy, loss = self._feedback_policy(cur_policy, real_time_delay, real_time_acc, macro)
        if self.correct_record_path:
            self.macro_search.pre_delay_and_acc(context_info, policy, record_path=self.correct_record_path)
        # Preserve the PR's hold policy near the constraint boundary.
        hold_policy = loss < 0.05
        if hold_policy:
            policy = copy.deepcopy(cur_policy)
        if self.steady_record_path:
            record = SteadyRecord(
                task_id=cur_task_id,
                **macro,
                delay_loss=self.micro_feedback.delay_loss,
                acc_loss=self.micro_feedback.acc_loss,
                coeff_info=self.micro_feedback.coeff_info,
                if_new_policy_is_bad=hold_policy,
                old_policy=cur_policy,
                new_policy=policy,
            )
            SteadyRecord.write_record(record, self.steady_record_path)
        return policy
