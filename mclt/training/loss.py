import abc
import random
from collections import defaultdict
from copy import deepcopy, copy
from itertools import chain
from math import log
from typing import Optional, List, Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp.grad_scaler import OptState
from torch.distributions import Normal

from mclt.data.base import TaskDefinition


class BaseMultiTaskLoss(nn.Module, abc.ABC):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
    ):
        super().__init__()
        self._tasks = tasks
        self._loss_funcs = {
            name: nn.BCEWithLogitsLoss() if task.multilabel else nn.CrossEntropyLoss()
            for name, task in tasks.items()
        }

    @abc.abstractmethod
    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        pass


class UniformWeightedLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(tasks)
        self._weights = weights or {task: 1.0 / len(tasks) for task in tasks}

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        loss = 0.0

        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            weight = self._weights[task_id]
            loss += self._loss_funcs[task_id](logits, labels) * weight

        return loss


class UncertaintyWeightedLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
    ):
        super().__init__(tasks)

        # exp(-lg_num_tasks) == 1.0 / len(tasks)
        self._log_squared_variances = nn.ParameterDict(
            {
                task: nn.Parameter(data=torch.tensor([0.0], requires_grad=True))
                for task in tasks
            }
        )

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        loss = 0.0

        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            log_sq_variance = self._log_squared_variances[task_id]
            inverted_variance = torch.exp(-log_sq_variance)
            loss += (
                self._loss_funcs[task_id](logits, labels) * inverted_variance
                + log_sq_variance
            )
        return loss


class RandomWeightedLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        distribution: Optional[torch.distributions.Distribution] = None,
    ):
        super().__init__(tasks)
        self._distribution = distribution or Normal(0.0, 1.0)

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        weights = self._distribution.sample((len(self._tasks),))
        weights = torch.softmax(weights, dim=-1)
        task_weights = {task: w for task, w in zip(self._tasks, weights)}

        loss = 0.0
        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            weight = task_weights[task_id]
            loss += self._loss_funcs[task_id](logits, labels) * weight
        return loss


class GradVaccineLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        beta: float = 1e-2,
    ):
        super().__init__(tasks)
        self._beta = beta
        self.task_num = len(tasks)
        self.rho_T = torch.zeros(self.task_num, self.task_num)

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        loss = 0.0
        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            loss += self._loss_funcs[task_id](logits, labels)
        return loss

    def backward(self, losses, **kwargs):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]
        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (
                        pc_grads[tn_i].norm() * grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm() * (
                            self.rho_T[tn_i, tn_j] * (1 - rho_ij ** 2).sqrt() - rho_ij * (
                                1 - self.rho_T[tn_i, tn_j] ** 2).sqrt()) / (
                                grads[tn_j].norm() * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt())
                    pc_grads[tn_i] += grads[tn_j] * w
                    batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j] = (1 - self._beta) * self.rho_T[tn_i, tn_j] + self._beta * rho_ij
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                param.grad.data = new_grads[beg:end].contiguous().view(
                    param.data.size()).data.clone()
            count += 1


class GradSurgeryLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
    ):
        super().__init__(tasks)

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        losses = {}
        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            losses[task_id] = self._loss_funcs[task_id](logits, labels)
        return losses

    def backward(self, losses, optimizer, scaler=None):
        grads_per_task, classifier_grads_per_loss = self._compute_gradients(losses, optimizer)

        if scaler is not None:
            inv_scale = 1. / scaler.get_scale()
            for grad in grads_per_task:
                grad.mul_(inv_scale)
            for grad in chain.from_iterable(classifier_grads_per_loss.values()):
                grad.mul_(inv_scale)

        # sprawdz skalę z grad checkp i bez

        if scaler is not None:
            # Unscale now so it won't do it before an optimization step
            scaler.unscale_(optimizer)

        flat_projected_grads = self._project_conflicting(grads_per_task)
        self._set_grad(flat_projected_grads, classifier_grads_per_loss, optimizer)

    @staticmethod
    def _project_conflicting(grads_per_task):
        num_tasks = len(grads_per_task)

        # use shallow copy and clone on demand
        projected_grads = copy(grads_per_task)
        cloned = torch.zeros(num_tasks, dtype=torch.bool)
        ignored = torch.tensor([(g.isnan()).any() for g in grads_per_task])

        # for i in range(num_tasks):
        #     index = [j for j in range(num_tasks) if i != j]
        #     random.shuffle(index)
        #     source_grads = projected_grads[i]
        #     for j in index:
        #         target_grads = grads_per_task[j]
        #         dot_product = torch.dot(source_grads, target_grads)
        #         if dot_product < 0:
        #             source_grads = source_grads if cloned[i] else source_grads.clone()
        #             cloned[i] = True
        #
        #             change = dot_product / target_grads.norm().pow(2)
        #             source_grads.div_(change).sub_(target_grads).mul_(change)
        #
        #             projected_grads[i] = source_grads

        # Save as much memory as possible by using a single tensor instead of stacking
        valid_projected_grads = [grads for grads, i in zip(projected_grads, ignored) if not i]
        if not valid_projected_grads:
            return projected_grads[0]

        accumulator = valid_projected_grads[0]
        for grad in valid_projected_grads[1:]:
            accumulator += grad
        return accumulator / num_tasks

    def _compute_gradients(self, losses, optimizer):
        grads_per_loss = []
        classifier_grads_per_loss = {}

        for loss in losses:
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)

            shared_grads, classifier_grads = self._retrieve_grad(optimizer)
            flat_grad = torch.cat([grad.flatten() for grad in shared_grads])
            grads_per_loss.append(flat_grad)
            for name, values in classifier_grads.items():
                assert name not in classifier_grads_per_loss
                classifier_grads_per_loss[name] = values

        return grads_per_loss, classifier_grads_per_loss

    @staticmethod
    def _retrieve_grad(optimizer):
        shared_grads = []
        classifier_grads = defaultdict(list)

        for group in optimizer.param_groups:
            if (name := group['name']) == 'backbone':
                for p in group['params']:
                    if p.grad is not None:
                        shared_grads.append(p.grad)
            else:
                for p in group['params']:
                    if p.grad is not None:
                        classifier_grads[name].append(p.grad.clone())

        return shared_grads, classifier_grads

    @staticmethod
    def _set_grad(flattened_grads, classifier_grads_per_loss, optimizer):
        for group in optimizer.param_groups:
            if (name := group['name']) == 'backbone':
                start = 0
                end = 0
                for p in group['params']:
                    if p.grad is not None:
                        end = start + p.grad.numel()
                        p.grad = flattened_grads[start:end].reshape(p.grad.shape)
                        start = end
                assert end == len(flattened_grads)

            else:
                if name not in classifier_grads_per_loss:
                    continue
                grads = classifier_grads_per_loss[name]
                params = group['params']
                assert len(grads) == len(params)
                for p, g in zip(params, grads):
                    p.grad = g
