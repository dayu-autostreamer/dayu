import os
import shutil
import tempfile


class FileOps:
    @staticmethod
    def save_data_file(task, file_data):
        file_path = task.get_file_path()
        with open(file_path, 'wb') as buffer:
            buffer.write(file_data)

    @staticmethod
    def remove_data_file(task):
        file_path = task.get_file_path()
        FileOps.remove_file(file_path)

    @staticmethod
    def remove_file(file_path):
        if not file_path or not os.path.exists(file_path):
            return

        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)

    @staticmethod
    def create_temp_directory(prefix):
        tmp_dir = os.path.join(tempfile.gettempdir(), prefix)
        FileOps.create_directory(tmp_dir)
        return tmp_dir

    @staticmethod
    def create_directory(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            assert os.path.isdir(dir_path), f'Path "{dir_path}" is a FILE'

    @staticmethod
    def clear_directory(dir_path):
        if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    @staticmethod
    def zip_directory(dir_path, zip_name, data_dir):
        """
        Create a .zip archive from the contents of `data_dir`, placing the archive in `dir_path`
        and naming it `zip_name`.

        Args:
            dir_path: Directory where the resulting .zip file will be stored.
            zip_name: Name of the .zip file to create (with or without the .zip extension).
            data_dir: Source directory whose contents will be archived.

        Returns:
            Absolute path to the created .zip file.
        """
        dir_path = os.path.abspath(dir_path)
        zip_path = os.path.join(dir_path, zip_name)
        base_name, _ = os.path.splitext(zip_path)
        shutil.make_archive(base_name, 'zip', root_dir=data_dir)
        return zip_path

    @staticmethod
    def unzip_file(zip_path, extract_dir):
        """
        Extracts a .zip file to the specified directory.

        Args:
            zip_path: Path to the .zip file to extract.
            extract_dir: Directory where the contents will be extracted.

        Returns:
            None
        """
        if not os.path.exists(zip_path) or not zip_path.lower().endswith('.zip'):
            raise ValueError(f'Invalid zip file path: {zip_path}')

        FileOps.create_directory(extract_dir)
        shutil.unpack_archive(zip_path, extract_dir, 'zip')
