import yaml

from mclt import PROJECT_PATH

PARAMS_PATH = PROJECT_PATH / 'params.yaml'


def load_config(*stage_name):
    with PARAMS_PATH.open('r') as f:
        config = yaml.safe_load(f)
        for name in stage_name:
            config = config[name]
        return config
