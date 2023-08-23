from .RepairStrategy import RepairStrategy
from .RepairWalk import RepairWalk
from .LearnableRepairWalk import LearnableRepairWalk
from ...params import RepairWalkParams


def repair_strategy_from_config(config,
                                cons_scoring_function=None,
                                var_scoring_function=None,
                                sample_indices=True):
    name = config['class']
    if name == 'LearnableRepairWalk':
        return LearnableRepairWalk(RepairWalkParams(),
                                   cons_scoring_function,
                                   var_scoring_function,
                                   sample_indices)
    else:
        raise NotImplementedError(f'repair strategy {name} not implemented')
