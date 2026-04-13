import os
import ast
from typing import Optional

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

        if not direct and isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                # Fallback to raw string if parsing fails
                pass

        return value

    @classmethod
    def _get_default_mount_path(cls) -> Optional[str]:
        default_mount_path = cls.get_parameter('DEFAULT_MOUNT_PATH')
        return os.path.normpath(default_mount_path) if default_mount_path else None

    @classmethod
    def _is_explicit_data_path(cls, file_path: str, default_mount_path: str, mount_prefix: str) -> bool:
        if not default_mount_path.startswith(mount_prefix + os.sep):
            return False

        default_root = os.path.relpath(default_mount_path, mount_prefix)
        return file_path == default_root or file_path.startswith(default_root + os.sep)

    @classmethod
    def get_default_file_path(cls) -> str:
        """
        Return the default directory mounted for the current component.

        Components that load local weights/configs by short relative names rely
        on `DEFAULT_MOUNT_PATH` to know where their primary asset directory is.
        """
        default_mount_path = cls._get_default_mount_path()
        if not default_mount_path:
            raise FileNotMountedError('Default file directory is not mounted.')
        return default_mount_path

    @classmethod
    def get_file_path(cls, file_path: str) -> str:
        """
        Resolve a runtime file reference.

        Callers can use two styles of references:
        1. default-relative names such as `retina_mnet.engine`, which resolve
           under `DEFAULT_MOUNT_PATH`;
        2. explicit container paths such as `scheduler/hei/reward.txt`, which
           resolve under `DATA_PATH_PREFIX` (or stay unchanged when absolute).

        The lookup prefers any existing explicit/default candidate first; if the
        file does not exist yet, we fall back to the default mount unless the
        caller already provided a path rooted at the default mount under
        `DATA_PATH_PREFIX`.
        """

        if os.path.isabs(file_path):
            return file_path

        mount_prefix = os.path.normpath(cls.get_parameter('DATA_PATH_PREFIX', '/home/data'))
        default_mount_path = cls.get_default_file_path()

        file_path = os.path.normpath(os.fspath(file_path))
        if file_path in ('', '.'):
            return default_mount_path

        data_prefix_candidate = os.path.join(mount_prefix, file_path)
        default_candidate = os.path.join(default_mount_path, file_path)

        for candidate in (data_prefix_candidate, default_candidate):
            if os.path.exists(candidate):
                return candidate

        if cls._is_explicit_data_path(file_path, default_mount_path, mount_prefix):
            return data_prefix_candidate
        return default_candidate

    @classmethod
    def get_temporary_file_path(cls, file_name: str) -> str:
        """
        Resolve the writable temporary directory used by runtime components.

        Temporary storage is always provided by the explicit temp mount and
        exposed through `TEMP_PATH`.
        """
        temp_dir = cls.get_parameter('TEMP_PATH')
        if not temp_dir:
            raise FileNotMountedError('Temporary directory is not mounted.')

        temp_dir = os.path.normpath(temp_dir)
        if not os.path.exists(temp_dir):
            raise FileNotMountedError(f"Temporary directory '{temp_dir}' is not mounted or does not exist.")

        temp_file_path = os.path.join(temp_dir, file_name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        return temp_file_path

    @classmethod
    def get_algorithm(cls, algorithm, al_name=None, **al_params):
        algorithm = algorithm.upper()

        algorithm_dict = Context.get_algorithm_info(algorithm, al_name, **al_params)
        if not algorithm_dict:
            return None
        return ClassFactory.get_cls(
            getattr(ClassType, algorithm),
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
            raise ValueError(f"Class '{class_name}' is not defined or imported.")
