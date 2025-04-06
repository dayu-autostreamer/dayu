import yaml
import os


class IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._root = os.path.split(stream.name)[0]

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, IncludeLoader)


class YamlOps:

    @staticmethod
    def read_yaml(yaml_file):
        IncludeLoader.add_constructor('!include', IncludeLoader.include)
        with open(yaml_file, 'r', encoding="utf-8") as f:
            values = yaml.load(f, Loader=IncludeLoader)
        return values

    @staticmethod
    def read_all_yaml(yaml_file):
        with open(yaml_file, 'r') as file:
            docs = list(yaml.safe_load_all(file))
        return docs

    @staticmethod
    def write_yaml(value_dict, yaml_file):
        with open(yaml_file, 'a', encoding="utf-8") as f:
            yaml.dump(data=value_dict, stream=f, encoding="utf-8", allow_unicode=True)

    @staticmethod
    def write_all_yaml(yaml_docs, yaml_file):
        with open(yaml_file, 'w') as file:
            yaml.safe_dump_all(yaml_docs, stream=file, encoding='utf-8', allow_unicode=True)

    @staticmethod
    def clean_yaml(yaml_file):
        with open(yaml_file, 'w') as f:
            f.truncate()

    @staticmethod
    def is_yaml_file(yaml_file):
        try:
            with open(yaml_file, 'r') as file:
                yaml.safe_load(file)
            return True
        except Exception as e:
            return False
