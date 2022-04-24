from mclt.training.loss import MultiTaskLoss, RandomWeightedLoss, UncertaintyWeightedLoss

LOSS_FUNCS = {
    'uniform': MultiTaskLoss,
    'random': RandomWeightedLoss,
    'uncertainty': UncertaintyWeightedLoss,
}
