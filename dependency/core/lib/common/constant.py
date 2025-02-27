from enum import Enum


class SystemConstant(Enum):
    DEFAULT = 'dayu'

    GENERATOR = 'generator'
    CONTROLLER = 'controller'
    SCHEDULER = 'scheduler'
    DISTRIBUTOR = 'distributor'
    PROCESSOR = 'processor'
    MONITOR = 'monitor'

    BACKEND = 'backend'
    FRONTEND = 'frontend'
    DATASOURCE = 'datasource'
    REDIS = 'redis'


class FileNameConstant(Enum):
    SCHEDULE_CONFIG = 'scheduler_config.yaml'
    DISTRIBUTOR_RECORD = 'record_data.db'


class NodeRoleConstant(Enum):
    CLOUD = 'cloud'
    EDGE = 'edge'
