import sys
from copy import deepcopy

from mclt import PROJECT_PATH
from mclt.data import TASK_LANGUAGE_TABLE
from mclt.utils.config import load_config
from mclt.utils.experiments import (
    create_datamodule,
    create_multilingual_model_trainer,
    run_adaptune_experiment,
)

sys.path.append(PROJECT_PATH)


config = load_config('train')
config.update(load_config(config['method']))
for lang_name, _ in TASK_LANGUAGE_TABLE.iterrows():
    config['tasks'] = None
    config['languages'] = lang_name
    for repeat in range(config['num_repeats']):
        config = deepcopy(config)
        config['model_random_state'] += repeat

        run_adaptune_experiment(
            config,
            create_datamodule=create_datamodule,
            create_model_trainer=create_multilingual_model_trainer,
            experiment_name='adaptune_multi_task',
        )
