import gurobipy as gp
from gurobipy import GRB

from src.rl.utils import TensorList, NoiseGenerator
from .LearningAlgorithm import LearningAlgorithm
from src.rl.mip import EnhancedModel


class EvolutionaryStrategies(LearningAlgorithm):
    _num_epochs: int
    _num_trajectories: int
    _learning_parameter: float

    _num_successes: int

    def __init__(self,
                 num_epochs: int,
                 num_trajectories: int,
                 learning_parameter: float):
        self._num_epochs = num_epochs
        self._num_trajectories = num_trajectories
        self._learning_parameter = learning_parameter
        self._num_successes = 0

    def train(self, fprl, instances):
        self._num_successes = 0
        policy_architecture = fprl.policy_architecture
        noise_generator = NoiseGenerator(policy_architecture.parameters())
        for epoch in range(self._num_epochs):
            gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
            for problem_instance in instances:
                gradient_estimate.add_to_self(self._train_instance_loop(fprl,
                                                                        problem_instance,
                                                                        noise_generator))
            gradient_estimate.scale(1 / len(instances))
            gradient_estimate.add_to_iterator(policy_architecture.parameters())

    def _train_instance_loop(self, fprl, instance, noise_generator):
        policy_architecture = fprl.policy_architecture
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture.gnn)
        for trajectory_num in range(self._num_trajectories):
            noise = noise_generator.sample()
            noise.scale(self._learning_parameter)
            noise.add_to_iterator(policy_architecture.parameters())
            fprl.find_solution(model)
            noise.scale(-1)
            noise.add_to_iterator(policy_architecture.parameters())
            if fprl.reward != 0:
                noise.scale(-pow(fprl.reward / self._learning_parameter, 2))
                gradient_estimate.add_to_self(noise)
                self._num_successes += 1
            model.reset()
        gradient_estimate.scale(1 / self._num_trajectories)
        return gradient_estimate
