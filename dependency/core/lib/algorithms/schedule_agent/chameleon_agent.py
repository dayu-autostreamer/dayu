import abc
import os

from core.lib.common import ClassFactory, ClassType, Context, LOGGER, EncodeOps
from core.lib.common import VideoOps
from core.lib.estimation import AccEstimator, OverheadEstimator
from core.lib.common import Queue, FileOps
from core.lib.network import NodeInfo, merge_address, NetworkAPIPath, NetworkAPIMethod, PortInfo, http_request
from core.lib.content import Task

from .base_agent import BaseAgent
import time

__all__ = ('ChameleonAgent',)

"""
Chameleon Agent Class

Implementation of Chameleon

Jiang J, Ananthanarayanan G, Bodik P, et al. Chameleon: scalable adaptation of video analytics[C]//Proceedings of the 2018 conference of the ACM special interest group on data communication. 2018: 253-266.

* Only suuport in http_video mode (need accuracy information)
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

        self.schedule_knobs = {'resolution': self.resolution_list,
                               'fps': self.fps_list}

        self.processor_address = None

        # 大周期、小周期(段)、计算best_num个配置所允许的最大耗时
        # 一个大周期里包含 profile_window / segment_size个段
        # 为一个新的大周期求best_num个配置的耗时不能超过calculate_time
        self.profile_window = profile_window
        self.segment_size = segment_size
        self.calculate_time = calculate_time

        # 每一个大周期需要选择的配置数
        self.best_num = best_num

        # best_config_list记录当前大周期已选好的best_num个配置,与其F1得分[(config,score),]
        # 初始化的时候从get_default_profile()中获取, 用于在冷启动的第一个大周期里确定best_num个配置
        self.best_config_list = self.get_default_profile()

        # 所有旋钮值的排列组合, 用于在每个大周期开头的第一个段里求出best_num个最优配置
        self.all_config_list = self.get_all_knob_combinations()

        # 默认的"黄金配置"(精度最高、消耗最高的那个), 用于计算F1 score来对配置进行选择和排序
        self.golden_config = {'resolution': self.resolution_list[-1], 'fps': self.fps_list[-1]}

        # 选择配置时所需的最新视频帧,一般数量也就几十帧
        self.raw_frames = Queue(maxsize=30)
        self.profiling_video_path = 'profiling_video.mp4'

        self.profiling_frames = []

        self.task_dag = None

        self.current_analytics = ''

        # 对F1_score进行排序时用到的阈值
        self.threshold = threshold

        self.acc_gt_dir = acc_gt_dir
        self.acc_estimator = None

        self.overhead_estimator = OverheadEstimator('Chameleon', 'scheduler/chameleon')

    def get_all_knob_combinations(self):
        all_config_list = []
        for resolution in self.resolution_list:
            for fps in self.fps_list:
                all_config_list.append({'resolution': resolution, 'fps': fps})
        return all_config_list

    # 用途：每一个大周期开始时，执行此函数，更新最优的best_num个配置
    # 注意: 该函数执行时间不能超过calculate_time
    def update_best_config_list_for_window(self):
        # 待填充列表，用于存储self.all_config_list中每一种配置组合的得分
        target_config_list = []

        # 为了比较各种配置旋钮的优劣，需要将其和黄金配置下的效果相比较
        # 由于配置空间很大，为每一种配置都重新根据raw_video计算F1十分困难，而且会导致性能严重下降, 所以采用了一个偷懒的方法
        # 即，为每一种配置的每一种取值打分；对于一种配置组合，将配置组合中每一个配置值对应的得分相乘，作为这个配置组合的得分，也就是config_score
        # 比如，给reso={360p, 480p, 720p} 分别打分0.8 0.9 0.95，给fps={10, 20,30}分别打分0.6 0.7 0.8，从而(360p, 10fps)的得分就是0.8*0.6=0.48
        # 这样一来就不用对配置空间中的每一种配置组合都分别计算F1分数，也可以间接比较不同配置的优劣了。

        # 具体的，如何为每一种配置的取值打分呢？
        # 答案是用这种取值代替黄金配置golden_config中的相应配置得到new_config，
        # 然后计算new_config在当前raw_video下ground_truth的F1 score
        # 由于new_config是基于黄金配置修改得到的，性能表现不会很差
        # 假设黄金配置是(720p, 30)，那么360p的得分就是(360p, 30)在raw_video下实际运行后与groud_truth相比得到的F1_score
        # 从而配置空间中某个普通配置(360p, 10)的配置得分就是(360p, 30)的F1分数与(720p, 10)的F1分数的乘积
        each_knob_score = {}  # 存储每一种配置的每一种取值的得分
        for knob in self.schedule_knobs:
            for value in self.schedule_knobs[knob]:
                new_config = self.golden_config
                new_config.update({knob: value})
                score = self.get_f1_score(new_config)
                each_knob_score[value] = score

        # 计算每一种配置组合的得分config_score
        for config in self.all_config_list:
            target_config_list.append((config, self.calculate_config_score(config, each_knob_score)))
        LOGGER.debug(f'[Config List] {target_config_list}')
        # 根据score为所有配置进行排序, 排序的时候过滤掉得分小于等于阈值的, 最后根据排序结果取最优的best_num个。
        # target_config_list中的每一个元素都是二元组，分别是配置以及得分
        target_config_list = [x for x in sorted(target_config_list, key=lambda x: x[1], reverse=True) if
                              x[1] > self.threshold][:self.best_num]

        # 提取 target_config_list中的配置，忽略配置的得分
        self.best_config_list = [x[0] for x in target_config_list]

    # 用途：从一个大周期的第二个segment开始，执行此函数，从已有的best_num个配置里选择一个最优的
    # 注意: 该函数执行时间不能超过calculate_time
    def update_best_config_list_for_segment(self):

        target_config_list = []
        each_knob_score = {}
        for knob in self.schedule_knobs:
            for value in self.schedule_knobs[knob]:
                new_config = self.golden_config
                new_config.update({knob: value})
                score = self.get_f1_score(new_config)
                each_knob_score[value] = score

        LOGGER.debug(f'[Knob Score] {each_knob_score}')

        # 为已有的best_num个配置打分
        for config in self.best_config_list:
            target_config_list.append((config, self.calculate_config_score(config, each_knob_score)))
        LOGGER.debug(f'[Config List] {target_config_list}')
        # 从已有的配置来选择最优的，不需要机械能筛选
        target_config_list = [x for x in sorted(target_config_list, key=lambda x: x[1], reverse=True)]

        # 得到重新排序的best_config_list
        self.best_config_list = [x[0] for x in target_config_list]

        # 为每一个配置计算分数，用于给配置排序

    @staticmethod
    def calculate_config_score(config, knob_value):
        res = 1
        # 遍历旋钮值
        for value in config.values():
            res *= knob_value[value]
        return res

    # 使用raw_data, 计算target_config相对于groud_truth的F1得分
    # 一般需要实际运行
    def get_f1_score(self, target_config):
        try:
            resolution = target_config['resolution']
            fps = target_config['fps']
            raw_resolution = VideoOps.text2resolution('1080p')
            resolution = VideoOps.text2resolution(resolution)
            resolution_ratio = (resolution[0] / raw_resolution[0], resolution[1] / raw_resolution[1])
            frames, hash_data = self.process_video(resolution, fps)
            LOGGER.debug(f'[FRAMES] length of frames:{len(frames)}')
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
        LOGGER.debug(f'[FRAMES] get from profiling frames num: {len(frames_info)}')
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
            processor_port = PortInfo.get_service_port(self.current_analytics)
            self.processor_address = merge_address(NodeInfo.hostname2ip(processor_hostname),
                                                   port=processor_port,
                                                   path=NetworkAPIPath.PROCESSOR_PROCESS_RETURN)

        cur_path = self.compress_video(frames)

        tmp_task = Task(source_id=0, task_id=0, source_device='', all_edge_devices=[], dag=self.task_dag)
        tmp_task.set_file_path(cur_path)
        response = http_request(url=self.processor_address,
                                method=NetworkAPIMethod.PROCESSOR_PROCESS_RETURN,
                                data={'data': tmp_task.serialize()},
                                files={'file': (tmp_task.get_file_path(),
                                                open(tmp_task.get_file_path(), 'rb'),
                                                'multipart/form-data')}
                                )
        FileOps.remove_data_file(tmp_task)
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
            LOGGER.info('[Fetch Frame] Fetch a frame from generator..')

        if not self.best_config_list:
            LOGGER.info('[No best config list] the length of best_config_list is 0!')
            return None

        policy = self.fixed_policy.copy()
        cloud_device = self.cloud_device

        self.current_analytics = self.task_dag.get_next_nodes('start')[0]
        for service_name in dag:
            dag[service_name]['service']['execute_device'] = cloud_device

        policy.update({'dag': dag})

        best_config = self.best_config_list[0]

        policy.update(best_config)
        return policy

    def run(self):
        # wait for video data
        while self.raw_frames.empty() or not self.current_analytics:
            time.sleep(1)

        LOGGER.info('[Chameleon Agent] Chameleon Agent Started')
        segment_num = 0  # 判断当前是第几个大周期里的第几个小周期了

        time.sleep(self.segment_size)

        while True:

            # profiling frames has a number lower bound
            if not self.raw_frames.full():
                continue

            segment_num += 1

            with self.overhead_estimator:
                self.profiling_frames = self.raw_frames.get_all_without_drop()

                # 冷启动时， 为初始化的best_num个配置排序
                if segment_num == 1:
                    self.update_best_config_list_for_segment()
                # 非冷启动且属于当前大周期开头时，重新选择best_num个配置并排序
                elif segment_num % (self.profile_window / self.segment_size) == 1:
                    self.update_best_config_list_for_window()
                # 非冷启动且不位于大周期开头时, 为已经选好的best_num个配置排序
                else:
                    self.update_best_config_list_for_segment()
            time_cost = self.overhead_estimator.get_latest_overhead()
            LOGGER.info(f'[Chameleon Profile] Profile for time: {time_cost}s')
            LOGGER.info(f'[Config List] Best Config List: {self.best_config_list}')
            if self.segment_size > time_cost:
                time.sleep(self.segment_size - time_cost)

    def get_default_profile(self):
        # extract from offline experiment
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
