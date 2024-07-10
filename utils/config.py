import yaml
from types import SimpleNamespace

def read_yaml_config(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)