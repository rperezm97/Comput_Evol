import numpy as np
from abc import ABC
import time
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm


class Evolutive_algorithm(ABC):
    """Abstract class that implements the methods of a generic evolutionary
    algorithm.

    Every child class of this implements a specific type of EA for a particular 
    problem, and it's meant to be organized in a subfolder in the current script 
    location.

    That folder should also contain an aux module with helper functions, aswell 
    as three folders: logs, parameters and plots."""

    # =============== INITIALIZATION OF VARIABLES AND PARAMETERS ===============
    def __init__(self,
                 instance_id):
        """Initialize the instance, the parameters, the population and their 
        fitness, and other relevant variabels for the algorithm."""

        # Instance_id includes information about the insatcne and the run,
        # In the format instanceName_runId. It's used for loading parameters,
        # logging, etc.
        self.instance_id = instance_id

        # IMPORTANT: When creating a child class, you need to define the base 
        # folder as self.base_folder=os.path.dirname(__file__) before calling 
        # this EA __init__. We need the base_folder it to load the instance, the
        # parameters, etc. 

        # LOAD INSTANCE AND PARAMETERS:
        self.load_instance()
        self.init_parameters()

        # INITIALIZE POPULATION VARIABLES:
        self.pop = self.init_pop()
        self.pop_fitness = self.f_fitness(self.pop)
        # Save the best individual with their adaptation value
        self.best = np.argmin(self.pop_fitness)
        self.best_adapt = self.pop_fitness[self.best]

    def load_instance(self):
        """Load the problem instance. To be implemented by child classes. 

        It should call aux.load_instance in each subclasses' auxiliary script, 
        so the child classes are problem-agnostic."""
        pass

    def init_parameters(self):
        """Load parameters instance from the json file corresponding to the 
        current instance_id"""

        # Parameters are stored in the "parameters" folder, in a json file named
        # after instanceName
        instanceName = "_".join(self.instance_id.split("_")[:2])
        parameters_file = os.path.join(self.base_folder,
                                       f"parameters/{instanceName}.json")
        # Laod the parameters dictinary and hold it as an attribute of the class
        with open(parameters_file) as f:
            parameters = json.load(f)
        # Use the keys in the parameter dict to define variables with the same
        # name (for example, self.n_pop = self.parameters["n_pop"])
        for key, value in parameters.items():
            setattr(self, key, value)
        print(parameters)

    # ====================ABTRACT METHODS OF A GENERIC EA======================-
    def init_pop(self):
        """Initialize the population. To be implemented by child classes."""
        pass

    def f_fitness(self):
        """Calculate the fitness value of one or more individual. It
        should work with np.arrays or list of individuals.
        To be implemented by child classes."""
        pass

    def parent_selection(self):
        """Select parents from the population. It should return a list with
        the indices of the matched parents, ready for variation operators.
        To be implemented by child classes."""
        pass

    def variation_operators(self, parents):
        """Perform the variation operators (crossover and mutation, but could 
        be others) on two parents to create two children. To be
        implemented by child classes."""
        pass

    def select_survivors(self, children):
        """Select the individuals to be included in the next generation.
        To be implemented by child classes."""
        pass

    # ========================== MAIN EXECUTION CYCLE ==========================

    def convergence_condition(self,gen,bests):
        """Check if the algorthm has converged (if the last quarter of the total 
        progress curve so far is a plateau)"""
        # We calculate the slope of teh curve, doin the discrete derivative
        # over 1/3 of the total n_gen
        
        change_rate=(bests[gen-self.n_gen//4]
                     - bests[gen]) / (bests[gen-self.n_gen//4])
        
        return change_rate<1e-5


    def evolve(self, early_stop=True):
        """
        Runs the evolutionary algorithm for ngen generations. In every iteration,
        it reproduces the population. Every ngen/100 iterations, it chekcs 
        convergence. If early_stop=True, it will stop when the convergence 
        condition has been archieved.

        Retuns the best and mean fitness values, with the standard deviation,
        of the 100 check generations.
        """

        # INITIALIZE VARIABLES:

        # Output variables
        means = np.zeros(self.n_gen)
        bests = np.zeros(self.n_gen)
        SDs = np.zeros(100)
        # Log
        log_file = os.path.join(self.base_folder,
                                f"logs/{self.instance_id}.txt")
        log = open(log_file, "w")
        # Convergence genration and time at the start of execution
        gen_conver = None
        t0 = time.time()

        # Check_step is the number of generations between convergence checkings.
        # It comes from dividing range(n_gen) in 100 equally spaced parts.
        check_step = (self.n_gen//100)

        print(f"Running {self.instance_id} for {self.n_gen} generations:\n")
        
        
        # MAIN EXECUTION CYCLE:
        for gen in tqdm(range(self.n_gen)):
            # Select parents and apply the variation operators
            parent_matches = self.parent_selection()
            children = self.variation_operators(parent_matches)
            # Select survivors and update the population variables (in place)
            self.select_survivors(children)


            # Store best fitness value and the the empirical mean over
            # the population
            bests[gen] = self.best_adapt
            means[gen] = np.sum(self.pop_fitness) / self.n_pop

            # If gen is a multiple of the check_step, we check convergence
            if not (gen % check_step):
                # Get the index of the current checking generation (from 1 to 100)
                checkgen_i = gen//check_step

                # Calculate and store standard deviation (pseudovariance) of the
                # population fitness, aswell as the best value
                SDs[checkgen_i] = np.sqrt(np.sum((self.pop_fitness
                                                  - means[gen])**2)
                                          / (self.n_pop - 1))

                # Print the progress update and write the info to the log file
                print((f"""  \n Gen {gen} :
                            \n Best = {bests[gen]} 
                            \n Mean = {means[gen]} 
                            \n STD = {SDs[checkgen_i]} 
                            \n \n """),
                      end="", flush=True)
                log.write(" ".join(map(str, [gen,
                                             bests[gen],
                                             means[gen],
                                             SDs[checkgen_i],
                                             "\n"])))

                # Check for convergence: in 100 generations, the best individual
                # hasn't improved (thus we have reached a plateau in exploration)
                # and the standard deviations hasn't changed (thus, the algorithm
                # has exploited that solution and t,e best individual has taken
                # over most of the population).
                if (gen>self.n_gen//4 and
                    self.convergence_condition(gen,
                                               bests=bests)):
                    if not gen_conver:                     
                        gen_conver = gen
                        if early_stop:
                            break

        # SAVING VALUES AND PLOTTING

        # Check the time of running
        t1 = time.time()
        t_conver = t1 - t0
        # Set the best and mean values of the generations after convergence as
        # the value of convergence, for the plots
        bests[gen:] = bests[gen]
        means[gen:] = means[gen]

        # Print the final results, formatting it accordingly to convergence
        # being reached or not
        print(("\n Convergence " +
               ("was" if gen_conver else "wasn't") +
               f""" reached in {t_conver} seconds, 
                    after {gen_conver or self.n_gen} 
                    generations.  
                    \n Fitness of the best individual: {self.best_adapt}"""),
              flush=True)
        # Write the convergence info in the log
        log.write(" ".join(map(str,
                           [gen_conver,
                            t_conver,
                            "\n"])))
        # Close log
        log.close()

        # Save the genome of the best individual, if save_best
        # TODO: log.write(str(self.pop[self.best]))

        # Plot the progress graph
        self.plot_convergence(gen_conver, bests, means, SDs)

        return self.pop[self.best], self.best_adapt, t_conver, gen_conver

    def plot_convergence(self, gen_converg, bests, means, SDs, 
                         visualize=False):
        """
        Plots the progress curve of the evolutive algorithm, including with the 
        value of  the best individual, the mean of the population, and the standard 
        deviation per generation.
        Also shows the point where the convergence condition is met (if ever).
        """

        generations = np.arange(self.n_gen)
        # Create a list with the indices of the 100 reference generations
        ref_generations = [i for i in range(
            self.n_gen) if not i % (self.n_gen//100)]
        # Create the figure
        plt.figure(figsize=(8, 6), dpi=200)
        plt.xlim(0, self.n_gen)
        if self.optimal:
            minimum = self.optimal
        else:
            minimum = min(bests)
        plt.ylim(minimum-0.05*max(bests), 1.05*max(bests))

        # Plot:
        # The mean

        plt.plot(generations,
                 means[:self.n_gen],
                 linewidth=.3,
                 color="orange",
                 label='Mean')
        # The standard deviation at the reference points
        plt.errorbar(ref_generations,
                     means[ref_generations],
                     yerr=SDs[:100*self.n_gen//self.n_gen],
                     color='Black',
                     elinewidth=.6,
                     capthick=.8,
                     capsize=.8,
                     alpha=.5,
                     fmt='o',
                     markersize=.7,
                     linewidth=.7)
        # The value of the best individual and the known optimal value
        plt.plot(generations,
                 bests[:self.n_gen],
                 "--",
                 linewidth=.8,
                 color="blue",
                 label='Best')

        if not (self.optimal is None):
            plt.plot(generations,
                     self.optimal*np.ones(self.n_gen),
                     label="Optimal",
                     color="green")
        # The convergence generation on the best individual curve
        if gen_converg:
            plt.scatter(gen_converg,
                   bests[gen_converg],
                    color="red",
                    label='Convergence',
                    zorder=1)

        # Add the grid, legend and labels

        plt.legend(loc='upper right',
                   frameon=False,
                   fontsize=8)
        plt.xlabel('Generations')
        plt.ylabel('Adaptation value')

        plt.title('{} '.format(self.instance_id))

        # Create the axis
        plt.grid(True)

        # Save the figure in the plots folder and show it
        plt.savefig(os.path.join(self.base_folder,
                                 'plots/{}.png'.format(self.instance_id)))

        # Show the plot
        if visualize:
            plt.show()
