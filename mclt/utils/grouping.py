from collections import defaultdict
from typing import Any

import torch
from torch import Tensor


def group_by_task(
    task: list[str],
    **kwargs,
) -> dict[str, Any]:
    input_to_task = defaultdict(lambda: defaultdict(list))
    for i, task_id in enumerate(task):
        for key, values in kwargs.items():
            input_to_task[task_id][key].append(values[i])
        input_to_task[task_id]['ids'].append(torch.tensor([i]))

    return {
        task_id: {k: torch.stack(v) for k, v in attrs.items()}
        for task_id, attrs in input_to_task.items()
    }
