from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        is_multilabel = len(features[0]['labels']) > 1

        if is_multilabel:
            max_len = max(len(item['labels']) for item in features)
            for item in features:
                labels = item['labels']
                labels = labels.tolist() + [-1] * (max_len - len(labels))
                item['labels'] = torch.tensor(labels)
        padded = super().__call__(
            [
                {
                    'input_ids': f['input_ids'],
                    'attention_mask': f['attention_mask'],
                    'labels': f['labels'],
                }
                for f in features
            ]
        )
        for key in features[0]:
            if key not in padded:
                padded[key] = [f[key] for f in features]

        return dict(padded)
