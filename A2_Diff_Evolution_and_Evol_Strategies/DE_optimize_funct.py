import numpy as np
import sys, os
import json
from scipy.stats.qmc import LatinHypercube
from multiprocessing import Pool, cpu_count

# Import additional modules and parent class
A2_FOLDER = os.path.dirname(__file__)
sys.path.append(A2_FOLDER)
import aux_DE as aux
import funct
sys.path.append( os.path.join(A2_FOLDER, ".."))
from EA import Evolutive_algorithm

class Differential_Evolution(Evolutive_algorithm):
    def __init__(self, instance_id):
        """
        Initialize the Differential Evolution algorithm.

        Parameters:
            instance_id (str): Identifier for the function to be optimized.
        """
        print("Initializing the Differential Strategies algorithm for function optimization\n INSTANCE: {}".format(instance_id))
        function_name = instance_id.split("_")[0]
        self.function = getattr(funct, function_name)()

        super().__init__(instance_id, A2_FOLDER, 0, convergence_thresholds=(0,0,1e-15))

    def init_parameters(self):
        """
        Placeholder for initialization of algorithm parameters.
        """
        pass

    def init_pop(self):
        """
        Initialize the population using Latin Hypercube Sampling (LHS).

        Returns:
            np.ndarray: Initialized population.
        """
        # Determine the domain diameter for rescaling
        diameter = (self.function.dom[1] - self.function.dom[0])

        # Initialize population using LHS
        sampler = LatinHypercube(d=self.n_dims)
        pop = (sampler.random(n=self.n_pop) - 0.5) * diameter

        return pop

    def f_fit(self, x):
        """
        Evaluate the objective function.

        Parameters:
            x (np.ndarray): A single individual or an array of individuals.

        Returns:
            float or np.ndarray: The evaluated objective function value(s), adjusted for the known optimum.
        """
        return self.function.evaluate(x) - self.function.optimal
    def select_parents(self, paradigm, null_differentials):
        """
        Parent selection for Differential Evolution.

        Parameters:
            paradigm (str): Either 'DE/rand/1/bin' or 'DE/best/1/bin'.
            null_differentials (bool): Whether to allow null differentials.

        Returns:
            np.ndarray: Parents selected for the next generation.
        """
        n_parents = 3 if paradigm == 'DE/best/1/bin' else 2

        # Pre-allocate parent array
        selected_parents = np.zeros((self.n_pop * n_parents, self.n_dims))

        for i in range(self.n_pop):
            # Randomly select two distinct indices from the population
            indices = np.random.choice(self.n_pop, size=2, replace=False)
            xp, xq = self.pop[indices]

            # Choose xr based on the paradigm
            if paradigm == 'DE/best/1/bin':
                xr = self.pop[np.argmin(self.f_fit(self.pop))]
            else:
                indices = np.random.choice(self.n_pop, size=1, replace=False)
                xr = self.pop[indices]

            if null_differentials and np.array_equal(xp, xq):
                # If null_differentials is True and xp and xq are the same,
                # differential will be zero; this is allowable.
                pass

            # Concatenate the parents for the next generation
            if n_parents == 3:
                selected_parents[i * n_parents: (i + 1) * n_parents] = np.vstack([xp, xq, xr])
            else:
                selected_parents[i * n_parents: (i + 1) * n_parents] = np.vstack([xp, xq])

        return selected_parents
    
    def variation_operators(self, parent_indices, use_jitter=False, use_dither=False):
        """
        Apply mutation and crossover (binomial) to generate offspring, with optional Jitter/Dither for F.

        Parameters:
            parent_indices (np.ndarray): Indices of selected parents.
            use_jitter (bool): Flag to activate Jitter.
            use_dither (bool): Flag to activate Dither.
            
        Returns:
            np.ndarray: Offspring population.
        """

        # Determine differential weight (F) based on flags
        if use_dither:
            F = np.random.uniform(0.5, 1.0)
        else:
            F = self.diff_weight
        
        if use_jitter:
            F += np.random.uniform(-0.1, 0.1, self.n_pop)  # Apply jitter per individual

        # Extract parent vectors based on indices.
        x_p, x_q = self.population[parent_indices[:, 0]], self.population[parent_indices[:, 1]]
        
        # Mutation
        differential = x_q - x_p  # Compute the differential vectors.
        donor_vectors = x_p + self.diff_weight * differential  # Generate donor vectors.

        # Crossover (binomial)
        n_dims = self.n_dims
        alpha_i = np.random.randint(1, n_dims + 1, self.n_pop)  # Randomly select one dimension for guaranteed mutation.
        mask = np.less(np.random.rand(self.n_pop, n_dims), self.crossover_prob)  # Create crossover mask.
        np.put_along_axis(mask, alpha_i[:, np.newaxis] - 1, 1, axis=1)  # Ensure alpha_i dimension is always mutated.
        
        # Generate offspring via binomial crossover.
        offspring = mask * donor_vectors + (1 - mask) * x_p

        # Bounce-back boundary handling.
        lower, upper = self.function.dom
        below_bound = offspring < lower
        above_bound = offspring > upper
        offspring[below_bound] = 2 * lower - offspring[below_bound]
        offspring[above_bound] = 2 * upper - offspring[above_bound]
    
        return offspring
    
    def selection(self, trial_vectors):
        """
        Apply deterministic selection based on the survival of the fittest.

        Parameters:
            trial_vectors (np.ndarray): Offspring (trial vectors) population.

        Returns:
            np.ndarray: New population for the next generation.
        """

        # Evaluate the fitness of trial vectors
        trial_fitness = self.f_fit(trial_vectors)

        # Evaluate the fitness of the current population (target vectors)
        current_fitness = self.f_fit(self.population)

        # Create a boolean mask indicating where the trial vectors are more fit
        mask_better = trial_fitness < current_fitness

        # Replace target vectors with trial vectors where the trial vectors are more fit
        self.population[mask_better] = trial_vectors[mask_better]

        return self.population
