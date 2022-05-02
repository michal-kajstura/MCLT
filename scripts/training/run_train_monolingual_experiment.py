from copy import deepcopy

from mclt.utils.config import load_config
from mclt.utils.experiments import create_baseline_model_trainer, create_datamodule, run_experiment

datasets = (
    # 'mtsc:pl',
    # 'mtsc:de',
    # 'mtsc:es',
    # 'polemo_in:pl',
    # 'amazon_reviews:en',
    # 'amazon_reviews:de',
    # 'amazon_reviews:es',
    # 'cyberbullying_detection:pl',
    # 'hate_speech_offensive:en',
    # 'semeval_2018:en',
    # 'semeval_2018:es',
    # 'go_emotions:en',
    # 'rureviews:ru',
    'xnli:ru',
)

config = load_config('train')
config.update(load_config(config['method']))

for dataset in datasets:
    config['datasets'] = [dataset]
    datamodule = create_datamodule(config)
    print(datamodule)
    for b in datamodule.train_dataloader():
        print(b)

    for repeat in range(config['num_repeats']):
        config = deepcopy(config)
        config['model_random_state'] += repeat

        run_experiment(
            config,
            model_trainer=create_baseline_model_trainer(
                config, datamodule.num_labels, datamodule.multilabel
            ),
            datamodule=datamodule,
            experiment_name=config['dataset'],
            experiment_tag=config['method'],
        )
