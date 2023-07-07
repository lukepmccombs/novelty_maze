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
from leap_qd.ns.ns import NoveltySearch
from leap_qd.ns.archive import NoveltySearchArchive
from leap_qd.ns.probe import NSDiversityPlot

from novelty_maze.leap.problem import MazeProblem
from novelty_maze.leap.probe import Scatter2DBehavior

import argparse
from distributed import Client
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SimpleNetwork(nn.Module):

    def __init__(self, num_inputs, hidden_layers, num_outputs, activation=F.leaky_relu):
        super().__init__()
        latent_sizes = [num_inputs, *hidden_layers, num_outputs]
        
        self.linear_layers = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(latent_sizes[:-1], latent_sizes[1:])
            ])
        self.activation = activation

    def forward(self, obs):
        latent = obs
        for layer in self.linear_layers[:-1]:
            latent = self.activation(layer(latent))

        return self.linear_layers[-1](latent)
        

def main(
            generations, pop_size,
            mutate_std, expected_num_mutations,
            model_hidden_layer_size, model_num_hidden_layer,
            n_workers=None
        ):
    device = "cpu"
    decoder = NumpyDecoder(device=device)
    
    ns = NoveltySearch(NoveltySearchArchive(5, 3 / 200, pop_size * 3)) 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot_probe = FitnessPlotProbe(
            modulo=1, ax=ax1
        )
    behav_probe = Scatter2DBehavior(ax2)
    div_probe = NSDiversityPlot(ns, (50, 50), (-1, 1), (-1, 1), ax3)
    
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
                    11, (model_hidden_layer_size,) * model_num_hidden_layer, 2,
                    activation=F.tanh
                ), decoder=decoder, individual_cls=DistributedEvaluationIndividual
            ),
            
            pipeline=[
                ns.assign_population_fitnesses,
                div_probe,
                behav_probe,
                plot_probe,
                ops.tournament_selection,
                ops.clone,
                mutate_gaussian(std=mutate_std, expected_num_mutations=expected_num_mutations),
                UniformCrossover(),
                *eval_pool_pipeline,
                ns.assign_population_fitnesses,
                ns.add_population_evaluations
            ],

            **({"init_evaluate": eval_population(client=client)} if n_workers is not None else {})
        )
    
    if n_workers is not None:
        client.close()
    
    plt.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-g", "--generations", default=500, type=int,
            help="The number of generations the evolutionary algorithm is ran for."
        )
    parser.add_argument(
            "-p", "--pop_size", default=100, type=int,
            help="The size of the population to be evolved."
        )
    parser.add_argument(
            "--model_hidden_layer_size", default=10, type=int,
            help="The number of hidden nodes per hidden layer in the individual models."
        )
    parser.add_argument(
            "--model_num_hidden_layer", default=3, type=int,
            help="The number of hidden layers in the individual models."
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