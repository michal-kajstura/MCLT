import os
os.environ['HF_DATASETS_OFFLINE ']= "1"

import pandas as pd

from tqdm import tqdm
from mclt.data import TASK_LANGUAGE_TABLE, DATASETS

stats = []
tokenizer = None

progress = tqdm(total=len(TASK_LANGUAGE_TABLE) * len(TASK_LANGUAGE_TABLE.columns))
for task_name, datasets in TASK_LANGUAGE_TABLE.iteritems():
    for lang_name, dataset in datasets.iteritems():
        progress.set_description(f'{task_name}-{lang_name}')
        progress.update()
        if dataset is None:
            continue

        datamodule_class = DATASETS[dataset]
        datamodule = datamodule_class(tokenizer, train_sample_size=None)
        datamodule.prepare_data()
        datamodule.setup()
        datasets = {attr: getattr(datamodule, attr) for attr in ('_train_dataset', '_val_dataset', '_test_dataset')}
        for k, v in datasets.items():
            stats.append({'dataset': dataset, 'language': lang_name, 'task': task_name, 'split': k, 'size': len(v)})

pd.DataFrame(stats).to_csv('../../notebooks/dataset_stats.csv', index=False)
