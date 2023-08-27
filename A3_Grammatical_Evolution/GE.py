import numpy as np
import sys, os
import json
from scipy.stats.qmc import LatinHypercube
from multiprocessing import Pool, cpu_count

# Import additional modules and parent class
A3_FOLDER = os.path.dirname(__file__)
sys.path.append(A3_FOLDER)
import aux as aux
import funct
sys.path.append( os.path.join(A3_FOLDER, ".."))
from EA import Evolutive_algorithm

import aux
from kernel import KG,KP,KS
class GrammaticalEvolution(Evolutive_algorithm):
    """
    Class that represents the Grammatical Evolution algorithm to solve a linear
    regression problem.
    The goal is to approximate a function f, given pairs of points (x_j, f(x_j)), 
    using a combination
    of Gaussian, Polynomial and Sigmoid kernels.
    """

    def __init__(self, instance_id, function, rules_file="rules.json"):
        """
        Initializes the Grammatical Evolution instance with the given filename 
        to load the parameters.
        """      
        print("\nInitializing the Gramatical Evolution algoritm for symboli regression \n-INSTANCE: {}\n".format(instance_id))
        
        function_name = instance_id.split("_")[0]
        self.function = getattr(funct, function_name)()
        
        super().__init__(instance_id, A3_FOLDER, None, convergence_thresholds=(0,0,1e-15))

        
        self.equidistant_sample= np.linspace(self.function.dom[0], 
                                            self.function.dom[0], 
                                            self.n_samples)

        
        super().__init__(instance_id, A2_FOLDER, 0, convergence_thresholds=(0,0,1e-15))

       
        
        # Get
        self.BNF_rules = aux.load_rules(rules_file)
        
    def init_pop(self):
        """
        Initializes the population for the Grammatical Evolution algorithm.

        It creates:
        - self.mask_codons: a 1D numpy array of length self.n_pop, containing the number of active codons for each individual.
        - self.population: a 2D numpy array of shape (self.n_pop, 8 * self.max_codons), filled with random binary values, where inactive codons are set to 0.
        """

        # Initialize the mask_codons array with random number of active codons
        self.mask_codons = np.random.randint(1, self.max_codons + 1, size=self.n_pop)

        # Initialize the population array with random binary numbers (0 or 1)
        self.pop = np.random.randint(0, 2, 
                                     size=(self.n_pop, 8 * self.max_codons), 
                                     dtype=np.uint8)

        # Set the inactive codons to 0 based on the mask_codons array
        for i in range(self.n_pop):
            self.population[i, 8 * self.mask_codons[i]:] = 0

    def f_fit(self, x):
        # The decoded function evaluated in the
        f_hat=aux.decode(x)
        
        # # Objective 1: Absolute mean ponderated error of the pairs 
        # (f_hat(x_j),f(x_j))
        f_error=abs(f_hat.f(self.sample_points)-self.f_sample_point)      
        omega = self.K0 if f_error <= self.U else self.K1
        f_error=(omega @ f_error)/self.m
        
        if self.penalty_weight:
            # Objective 2: Penalty for the number of kernels
            n_kernels=len(f_hat.kernels)
            kernel_penalty=self.kernel_penalty_weight * n_kernels
            return f_error + kernel_penalty
        else:
            return f_error
    
    def parent_selection(self):
        """
        Select parents for the next generation using tournaments.
        """
        parents_idx = [aux.tournament_selection(self.pop_fit, self.t_size)
                       for _ in range(self.n_tournaments)]
        if self.n_pop % 2:
            parents_idx.append(None)
        return np.array(parents_idx)

    def match_parents(self, parents):
        """
        Match the parents 2 by 2 randomly and return a 3d array to index
        """
        return parents.reshape(-1, 2)

    def crossover(self, parents):
        """
        Perform single-point crossover on the parents.
        """
        p1, p2 = parents
        # If crossover probability is not met or second parent is None, 
        # return parents
        if (random.random() >= self.pc or p2 is None):
            return p1, p2

        # Choose a random crossover point
        cross_point = np.random.randint(1, len(p1)//15)

        # Create two empty children
        c1 = np.zeros(len(p1), dtype=int)
        c2 = np.zeros(len(p2), dtype=int)

        # Perform the crossover
        c1[:cross_point], c1[cross_point:] = p1[:cross_point], p2[cross_point:]
        c2[:cross_point], c2[cross_point:] = p2[:cross_point], p1[cross_point:]

        return c1, c2

    def mutate(self, x):
        """
        Perform a mutation on individual x by swapping two randomly chosen genes.
        """
         # If the random number generated is greater than the mutation probability,
        # return the original individual
        if random.random() >= self.pm:
            
            return x
        # Get the length of the individual
        n = len(x)
        # Randomly select two genes to be swapped
        idx = np.random.randint(0, n-1)
        # Swap the two selected genes
        x[idx] = random.randint(0,255)
        
        i, = np.where(self.pop == x)
        kernel_start=(i-i%15)
        kernel_position=kernel_start//15
        
        changed_kernel=aux.decode[x[kernel_start:kernel_start+15]]
        self.pop_decoded[i].kernels[kernel_position]=changed_kernel.kernels[0]
        self.pop_decoded[i].weights[kernel_position]=changed_kernel.weights[0]
        self.pop_adapts[i]=self.f_adapt(x)
        
        return x

    def select_survivors(self, children):
        """
        Substitute the previous popualtion with the children and apply elitism
        if applicable.
        """
        # Save the previous best individual for elitism and update the
        # population with the children
        prev_best_adapt=self.best_adapt
        prev_best=self.pop[self.best]
        self.pop=children
        self.pop_fit = self.f_adapt(children)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)
        # If we apply elitism and the best individual of the previous generation
        # is better than the best individual of this generation
        if self.elitism and  self.best_adapt>prev_best_adapt:
            # Subtitute the worst individual of this generation with the 
            # best of the previous generation
            worst=np.argmax(self.pop_fit)
            self.pop[worst]=prev_best
            self.pop_fit[worst]=prev_best_adapt
            self.best_adapt=prev_best_adapt