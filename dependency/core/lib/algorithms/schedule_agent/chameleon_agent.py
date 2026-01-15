import abc
import os

from core.lib.common import ClassFactory, ClassType, Context, LOGGER, EncodeOps, VideoOps, Queue, FileOps, TaskConstant
from core.lib.estimation import AccEstimator, OverheadEstimator
from core.lib.network import NodeInfo, merge_address, NetworkAPIPath, NetworkAPIMethod, PortInfo, http_request
from core.lib.content import Task

from .base_agent import BaseAgent
import time

__all__ = ('ChameleonAgent',)

"""
Chameleon Agent Class

Implementation of Chameleon

Jiang J, Ananthanarayanan G, Bodik P, et al. Chameleon: scalable adaptation of video analytics[C]//Proceedings of the 2018 conference of the ACM special interest group on data communication. 2018: 253-266.

* Only support in http_video mode (needs accuracy information)
"""


@ClassFactory.register(ClassType.SCH_AGENT, alias='chameleon')
class ChameleonAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, fixed_policy: dict, acc_gt_dir: str,
                 best_num: int = 5, threshold=0.1,
                 profile_window=16, segment_size=4, calculate_time=1):
        super().__init__(system, agent_id)

        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.schedule_plan = None

        self.fixed_policy = fixed_policy

        self.fps_list = system.fps_list.copy()
        self.resolution_list = system.resolution_list.copy()

        self.local_device = NodeInfo.get_local_device()

        self.schedule_knobs = {'resolution': self.resolution_list,
                               'fps': self.fps_list}

        self.processor_address = None


        # Windowing configuration:
        # - A large profiling window contains (profile_window / segment_size) segments
        # - Computing the best_num configs at the start of a new window must finish within calculate_time
        self.profile_window = profile_window
        self.segment_size = segment_size
        self.calculate_time = calculate_time

        # Number of configurations to select per large window
        self.best_num = best_num

        # best_config_list stores the selected best_num configs for the current window with their F1 scores [(config, score),]
        # Initialized from get_default_profile() for cold start to populate the first window
        self.best_config_list = self.get_default_profile()

        # Cartesian product of all knob values, used to search top best_num configs at the first segment of each window
        self.all_config_list = self.get_all_knob_combinations()

        # Default "golden configuration" (highest accuracy and cost), used for computing F1 scores to rank configs
        self.golden_config = {'resolution': self.resolution_list[-1], 'fps': self.fps_list[-1]}

        # Recent frames used during profiling and selection (usually a few dozen)
        self.raw_frames = Queue(maxsize=30)
        self.profiling_video_path = 'profiling_video.mp4'

        self.profiling_frames = []

        self.task_dag = None

        self.current_analytics = ''

        # Threshold used when ranking by F1 score
        self.threshold = threshold

        self.acc_gt_dir = acc_gt_dir
        self.acc_estimator = None

        self.overhead_estimator = OverheadEstimator('Chameleon', 'scheduler/chameleon', agent_id=self.agent_id)

    def get_all_knob_combinations(self):
        all_config_list = []
        for resolution in self.resolution_list:
            for fps in self.fps_list:
                all_config_list.append({'resolution': resolution, 'fps': fps})
        return all_config_list

    # Purpose: At the beginning of each large window, update the top best_num configurations.
    # Note: This function must complete within calculate_time.
    def update_best_config_list_for_window(self):
        # Buffer to store the score for every combination in self.all_config_list
        target_config_list = []

        # To compare different knob settings, we compare them against the golden configuration.
        # The configuration space can be large; recomputing F1 for every combination on raw video is expensive.
        # Instead, we score each individual knob value by substituting it into the golden config to form new_config
        # and computing its F1 on the current raw video. Then, for a config combination, we multiply the scores of
        # its constituent knob values to approximate the combination's score (config_score).
        # Example: if reso={360p, 480p, 720p} have scores 0.8, 0.9, 0.95 and fps={10, 20, 30} have scores 0.6, 0.7, 0.8,
        # then the score for (360p, 10fps) is 0.8 * 0.6 = 0.48.

        # Specifically, scoring each knob value is done by replacing the corresponding field in golden_config to get
        # new_config, then computing the F1 score against the ground truth on the current raw video. Because new_config
        # is close to the golden config, performance is acceptable.
        # For example, if golden is (720p, 30fps), then the score of 360p is the F1 of (360p, 30fps) on the raw video.
        # The score of a general config like (360p, 10fps) is the product of the F1 of (360p, 30fps) and (720p, 10fps).
        each_knob_score = {}  # Stores the score for every value of each knob
        for knob in self.schedule_knobs:
            for value in self.schedule_knobs[knob]:
                new_config = self.golden_config.copy()
                new_config.update({knob: value})
                score = self.get_f1_score(new_config)
                each_knob_score[value] = score

        # Compute config_score for every combination
        for config in self.all_config_list:
            target_config_list.append((config, self.calculate_config_score(config, each_knob_score)))
        LOGGER.debug(f'[Config List] {target_config_list}')
        # Sort by score in descending order, filter out those with score <= threshold, and take the top best_num
        # Each element of target_config_list is a tuple: (config, score)
        target_config_list = [x for x in sorted(target_config_list, key=lambda x: x[1], reverse=True) if
                              x[1] > self.threshold][:self.best_num]

        # Extract the configs only and drop the scores
        self.best_config_list = [x[0] for x in target_config_list]

    # Purpose: From the second segment of a window onward, rank and select among the existing best_num configs.
    # Note: This function must complete within calculate_time.
    def update_best_config_list_for_segment(self):

        target_config_list = []
        each_knob_score = {}
        for knob in self.schedule_knobs:
            for value in self.schedule_knobs[knob]:
                new_config = self.golden_config.copy()
                new_config.update({knob: value})
                score = self.get_f1_score(new_config)
                each_knob_score[value] = score

        LOGGER.debug(f'[Knob Score] {each_knob_score}')

        # Score the existing best_num configs
        for config in self.best_config_list:
            target_config_list.append((config, self.calculate_config_score(config, each_knob_score)))
        LOGGER.debug(f'[Config List] {target_config_list}')
        # Choose the best among existing configs; no additional filtering
        target_config_list = [x for x in sorted(target_config_list, key=lambda x: x[1], reverse=True)]

        # Update the reordered best_config_list
        self.best_config_list = [x[0] for x in target_config_list]

        # Compute a score for each config to enable ranking

    @staticmethod
    def calculate_config_score(config, knob_value):
        res = 1
        # Iterate over knob values
        for value in config.values():
            res *= knob_value[value]
        return res

    # Use raw data to compute the F1 score of target_config relative to the ground truth.
    # Typically requires actually running the analytics.
    def get_f1_score(self, target_config):
        try:
            resolution = target_config['resolution']
            fps = target_config['fps']
            raw_resolution = VideoOps.text2resolution('1080p')
            resolution = VideoOps.text2resolution(resolution)
            resolution_ratio = (resolution[0] / raw_resolution[0], resolution[1] / raw_resolution[1])
            frames, hash_data = self.process_video(resolution, fps)
            LOGGER.debug(f'[FRAMES] length of frames: {len(frames)}')
            results = self.execute_analytics(frames)
            # LOGGER.debug(f'[Analysis results] {results}')
            LOGGER.debug(f'[Hash codes] {hash_data}')
            if not self.acc_estimator:
                self.create_acc_estimator()
            acc = self.acc_estimator.calculate_accuracy(hash_data, results, resolution_ratio, fps / 30)
        except Exception as e:
            LOGGER.warning(f'Calculate accuracy failed: {str(e)}')
            acc = 0
        return acc

    def create_acc_estimator(self):
        if not self.current_analytics:
            raise ValueError('No value of "current_analytics" has been set')
        gt_path_prefix = os.path.join(self.acc_gt_dir, self.current_analytics)
        gt_file_path = Context.get_file_path(os.path.join(gt_path_prefix, 'gt_file.txt'))
        LOGGER.debug(f'[ACC GT] gt file path: {gt_file_path}')
        self.acc_estimator = AccEstimator(gt_file_path)

    def process_video(self, resolution, fps):
        import cv2
        raw_fps = 30
        fps = min(fps, raw_fps)
        fps_mode, skip_frame_interval, remain_frame_interval = self.get_fps_adjust_mode(raw_fps, fps)

        frame_count = 0
        frame_list = []
        frames_info = self.profiling_frames.copy()
        LOGGER.debug(f'[FRAMES] number of profiling frames: {len(frames_info)}')
        new_frame_hash_codes = []
        for frame, hash_code in frames_info:
            frame_count += 1
            if fps_mode == 'skip' and frame_count % skip_frame_interval == 0:
                continue

            if fps_mode == 'remain' and frame_count % remain_frame_interval != 0:
                continue
            frame = cv2.resize(frame, resolution)
            frame_list.append(frame)
            new_frame_hash_codes.append(hash_code)

        return frame_list, new_frame_hash_codes

    def execute_analytics(self, frames):
        if not self.processor_address:
            processor_hostname = NodeInfo.get_cloud_node()
            processor_port = PortInfo.get_service_port(self.local_device, self.current_analytics)
            self.processor_address = merge_address(NodeInfo.hostname2ip(processor_hostname),
                                                   port=processor_port,
                                                   path=NetworkAPIPath.PROCESSOR_PROCESS_RETURN)

        cur_path = self.compress_video(frames)

        tmp_task = Task(source_id=-1, task_id=-1, source_device='', all_edge_devices=[], dag=self.task_dag)
        tmp_task.set_file_path(cur_path)
        response = http_request(url=self.processor_address,
                                method=NetworkAPIMethod.PROCESSOR_PROCESS_RETURN,
                                data={'data': tmp_task.serialize()},
                                files={'file': (tmp_task.get_file_path(),
                                                open(tmp_task.get_file_path(), 'rb'),
                                                'multipart/form-data')}
                                )
        FileOps.remove_file(tmp_task.get_file_path())
        if response:
            task = Task.deserialize(response)
            return task.get_first_content()
        else:
            return None

    @staticmethod
    def get_fps_adjust_mode(fps_raw, fps):
        skip_frame_interval = 0
        remain_frame_interval = 0
        if fps >= fps_raw:
            fps_mode = 'same'
        elif fps < fps_raw // 2:
            fps_mode = 'remain'
            remain_frame_interval = fps_raw // fps
        else:
            fps_mode = 'skip'
            skip_frame_interval = fps_raw // (fps_raw - fps)

        return fps_mode, skip_frame_interval, remain_frame_interval

    def compress_video(self, frames):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = frames[0].shape
        video_path = self.profiling_video_path
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

        return video_path

    def get_schedule_plan(self, info):

        frame_encoded = info['frame']
        frame_hash_code = info['hash_code']
        dag = info['dag']

        self.task_dag = Task.extract_dag_from_dag_deployment(dag)

        if frame_encoded:
            self.raw_frames.put((EncodeOps.decode_image(frame_encoded), frame_hash_code))
            LOGGER.info('[Fetch Frame] Fetch a frame from generator.')

        if not self.best_config_list:
            LOGGER.info('[No best config list] the length of best_config_list is 0!')
            return None

        policy = self.fixed_policy.copy()
        cloud_device = self.cloud_device

        self.current_analytics = self.task_dag.get_next_nodes(TaskConstant.START.value)[0]
        for service_name in dag:
            dag[service_name]['service']['execute_device'] = cloud_device

        policy.update({'dag': dag})

        best_config = self.best_config_list[0]

        policy.update(best_config)
        return policy

    def run(self):
        # Wait for video data
        while self.raw_frames.empty() or not self.current_analytics:
            time.sleep(1)

        LOGGER.info('[Chameleon Agent] Chameleon Agent Started')
        segment_num = 0  # Track which segment within which large window

        time.sleep(self.segment_size)

        while True:

            # Profiling requires at least a certain number of frames
            if not self.raw_frames.full():
                continue

            segment_num += 1

            with self.overhead_estimator:
                self.profiling_frames = self.raw_frames.get_all_without_drop()

                # Cold start: rank the initial best_num configs
                if segment_num == 1:
                    self.update_best_config_list_for_segment()
                # Non-cold start and at the beginning of a new large window: select and rank best_num configs
                elif segment_num % (self.profile_window / self.segment_size) == 1:
                    self.update_best_config_list_for_window()
                # Non-cold start and not at a window boundary: re-rank existing best_num configs
                else:
                    self.update_best_config_list_for_segment()
            time_cost = self.overhead_estimator.get_latest_overhead()
            LOGGER.info(f'[Chameleon Profile] Profile for time: {time_cost}s')
            LOGGER.info(f'[Config List] Best Config List: {self.best_config_list}')
            if self.segment_size > time_cost:
                time.sleep(self.segment_size - time_cost)

    def get_default_profile(self):
        # Extracted from offline experiments
        return [{'resolution': '1080p', 'fps': 30},
                {'resolution': '1080p', 'fps': 10},
                {'resolution': '900p', 'fps': 30},
                {'resolution': '900p', 'fps': 10},
                {'resolution': '720p', 'fps': 30}]

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
