import os
from typing import Union

from .class_factory import ClassFactory, ClassType
from .error import FileNotMountedError


class Context:
    """The Context provides the capability of obtaining the context"""
    parameters = os.environ

    @classmethod
    def get_parameter(cls, param, default=None, direct=True):
        """get the value of the key `param` in `PARAMETERS`,
        if not exist, the default value is returned"""

        value = cls.parameters.get(param) or cls.parameters.get(str(param).upper())
        value = value if value else default

        if not direct:
            value = eval(value)

        return value

    @classmethod
    def get_file_path(cls, file_path: Union[str, int]) -> str:
        """
        Returns the full mount path for a given file path or volume index.

        If `file_path` is a string, the function searches for the specified file within the mounted volumes and returns
        the full path if the file is found. If `file_path` is an integer, it returns the path to the corresponding
        volume based on the index provided.

        Args:
            file_path (Union[str, int]): File path (str) or volume index (int).

        Returns:
            str: The full path to the file within the volume or the volume path itself.

        Raises:
            FileNotMountedError: If the specified file is not found in any mounted volumes.
            IndexError: If `file_path` is an integer but out of the valid volume index range.
        """
        volume_num = cls.get_parameter('VOLUME_NUM', direct=False)
        file_prefix = os.path.normpath(cls.get_parameter('FILE_PREFIX', ''))
        mount_prefix = os.path.normpath(cls.parameters.get('DATA_PATH_PREFIX', '/home/data'))

        # if input file path is integer, return corresponding volume path (volume0)
        if isinstance(file_path, int):
            if 0 <= file_path < volume_num:
                return os.path.join(mount_prefix, f'volume{file_path}')
            else:
                raise IndexError(f"Volume index {file_path} is out of range for {volume_num} volumes.")

        file_path = os.path.normpath(file_path)
        if volume_num <= 0:
            raise FileNotMountedError('No file directory is mounted')
        elif volume_num == 1:
            raw_file_path = cls.get_parameter(f'VOLUME_0')
            related_raw_dir = str(os.path.relpath(raw_file_path, file_prefix)) \
                if raw_file_path.startswith(file_prefix) else ''
            if file_path.startswith(raw_file_path):
                file_name = os.path.relpath(file_path, raw_file_path)
            elif file_path.startswith(related_raw_dir):
                file_name = os.path.relpath(file_path, related_raw_dir)
            elif not os.path.isabs(file_path):
                file_name = file_path
            else:
                raise FileNotMountedError(f"File '{file_path}' is not mounted.")
            return os.path.join(mount_prefix, 'volume0', file_name)
        else:
            for index in range(volume_num):
                raw_file_path = cls.get_parameter(f'VOLUME_{index}')
                related_raw_dir = str(os.path.relpath(raw_file_path, file_prefix)) \
                    if raw_file_path.startswith(file_prefix) else ''
                if file_path.startswith(raw_file_path):
                    file_name = os.path.relpath(file_path, raw_file_path)
                elif file_path.startswith(related_raw_dir):
                    file_name = os.path.relpath(file_path, related_raw_dir)
                else:
                    continue
                return os.path.join(mount_prefix, f'volume{index}', file_name)

            raise FileNotMountedError(f"File '{file_path}' is not mounted.")

    @classmethod
    def get_algorithm(cls, algorithm, al_name=None, **al_params):
        algorithm = algorithm.upper()

        algorithm_dict = Context.get_algorithm_info(algorithm, al_name, **al_params)
        if not algorithm_dict:
            return None
        return ClassFactory.get_cls(
            eval(f'ClassType.{algorithm}'),
            algorithm_dict['method']
        )(**algorithm_dict['param'])

    @classmethod
    def get_algorithm_info(cls, algorithm, name, **param):
        al_name = cls.get_parameter(f'{algorithm}_NAME') if name is None else name
        al_params = cls.get_parameter(f'{algorithm}_PARAMETERS', default='{}', direct=False)

        if not al_name:
            return None

        al_params.update(**param)

        algorithm_dict = {
            'method': al_name,
            'param': al_params
        }

        return algorithm_dict

    @classmethod
    def get_instance(cls, class_name, **instance_params):
        if class_name in globals():
            params = cls.get_parameter(f'{class_name.upper()}_PARAMETERS', default='{}', direct=False)
            instance_params.update(params)
            instance = globals()[class_name](**instance_params)
            return instance
        else:
            assert None, f"Class '{class_name}' is not defined or imported."


