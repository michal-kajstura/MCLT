from copy import deepcopy

from mclt.utils.config import load_config
from mclt.utils.experiments import (
    create_datamodule,
    create_multilingual_model_trainer,
    run_experiment,
)

config = load_config('train')
config.update(load_config(config['method']))

datamodule = create_datamodule(config)

for repeat in range(config['num_repeats']):
    config = deepcopy(config)
    config['model_random_state'] += repeat

    run_experiment(
        config,
        model_trainer=create_multilingual_model_trainer(config, datamodule.tasks),
        datamodule=datamodule,
        experiment_name=datamodule.name,
        experiment_tag=config['method'],
    )
