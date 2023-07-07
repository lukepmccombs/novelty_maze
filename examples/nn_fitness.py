from leap_torch.ops import mutate_gaussian, UniformCrossover
from leap_torch.initializers import create_instance
from leap_torch.decoders import NumpyDecoder

from leap_ec import ops
from leap_ec.probe import FitnessPlotProbe
from leap_ec.distrib.synchronous import eval_pool, eval_population
from leap_ec.representation import Representation
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep.executable import ArgmaxExecutable, WrapperDecoder

from leap_qd.individual import DistributedEvaluationIndividual
from leap_qd.ops import assign_population_fitnesses

from novelty_maze.problem import MazeProblem

import argparse
from distributed import Client
import matplotlib.pyplot as plt
from torch import nn

class SimpleNetwork(nn.Module):

    def __init__(self, num_inputs, hidden_layers, num_outputs, activation=F.relu):
        super().__init__()
        latent_sizes = [num_inputs, *hidden_layers, num_outputs]
        
        self.linear_layers = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(latent_sizes[:-1], latent_sizes[1:])
            ])
        self.activation = activation

    def forward(self, obs):
        latent = obs
        for layer in self.linear_layers:
            latent = self.activation(layer(latent))

        return latent
        

def main(
            generations=250, pop_size=30,
            hidden_nodes=10, hidden_layers=1,
            mutate_std=0.05, expected_num_mutations=1,
            n_workers=None
        ):
    device = "cpu"
    decoder = NumpyDecoder(device=device)
        
    plot_probe = FitnessPlotProbe(
            ylim=(0, 1), xlim=(0, 1),
            modulo=1, ax=plt.gca()
        )
    
    eval_pool_pipeline = [
            ops.evaluate,
            ops.pool(size=pop_size)
        ]
    
    if n_workers is not None:
        client = Client(n_workers=n_workers)
        eval_pool_pipeline = [eval_pool(client=client, size=pop_size)]
    
    generational_ea(
            max_generations=generations, pop_size=pop_size,
            
            problem=MazeProblem(),
            
            representation=Representation(
                initialize=create_instance(
                    SimpleNetwork,
                    11, (hidden_nodes,) * hidden_layers, 2
                ), decoder=decoder, individual_cls=DistributedEvaluationIndividual
            ),
            
            pipeline=[
                assign_population_fitnesses(func=lambda x: x["quality"]),
                ops.tournament_selection,
                ops.clone,
                mutate_gaussian(std=mutate_std, expected_num_mutations=expected_num_mutations),
                UniformCrossover(),
                *eval_pool_pipeline,
                assign_population_fitnesses(func=lambda x: x["quality"]),
                plot_probe,
            ],

            **({"init_evaluate": eval_population(client=client)} if n_workers is not None else {})
        )
    
    if n_workers is not None:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-g", "--generations", default=250, type=int,
            help="The number of generations the evolutionary algorithm is ran for."
        )
    parser.add_argument(
            "-p", "--pop_size", default=30, type=int,
            help="The size of the population to be evolved."
        )
    parser.add_argument(
            "--hidden_nodes", default=10, type=int,
            help="The number of hidden nodes per hidden layer in the model."
        )
    parser.add_argument(
            "--hidden_layers", default=1, type=int,
            help="The number of hidden layers in the model."
        )
    parser.add_argument(
            "--mutate_std", default=0.05, type=float,
            help="The standard deviation of the mutation applied to the genomes."
        )
    parser.add_argument(
            "--expected_num_mutations", default=1, type=int,
            help="The expected number of mutations to occur within the genome."
        )
    parser.add_argument(
            "--n_workers", default=None, type=int,
            help="If set, uses distributed evaluation of the poplation with the given number of workers."
        )
    
    args = parser.parse_args()
    main(**args.__dict__)