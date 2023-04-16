import numpy as np
from abc import ABC
import time
import matplotlib.pyplot as plt

import os

class Evolutive_algorithm(ABC):
    """Abstract class that implements the methods of a generic evolutionary
    algorithm and serves as a parent for other algorithms."""

    def __init__(self, name):
        """Initialize the population, calculate the adaptation of each
        individual, and store the best individual and its adaptation."""
        self.name = name
        self.pop = self.init_pop()
        self.pop_fit = self.f_adapt(self.pop)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)
        self.optimal=None
        self.parameters= None
    def init_pop(self):
        """Initialize the population. To be implemented by child classes."""
        pass

    def f_adapt(self, pop):
        """Calculate the adaptation of each individual in the population.
        To be implemented by child classes."""
        pass

    def parent_selection(self):
        """Select parents from the population. It should return a list with
        the indices of the parents. To be implemented by child 
        classes."""
        pass

    def match_parents(self, parents_idx):
        """Match the selected parents for crossover. It should return a list 
        with tuples of indices of the matched parents. 
        To be implemented by child classes."""

    def crossover(self, parents):
        """Perform crossover on two parents to create two children. To be
        implemented by child classes."""
        pass

    def mutate(self, individual):
        """Perform mutation on an individual. To be implemented by 
        child classes."""
        pass

    def select_survivors(self, parents, children):
        """Select the individuals to be included in the next generation.
        To be implemented by child classes."""
        pass

    def reproduce(self):
        """Perform reproduction for 1 generations."""
        # Select parents and create the matches by reshape the parent indices
        # array into a 3D array (for indexing the population matrix)
        parents_idx = self.parent_selection()
        # Reshape the parent indices array into a 3D array for indexing the
        # population matrix
        parent_matches = self.match_parents(parents_idx)

        children = np.empty((self.n_children, self.pop.shape[1]))
        i = 0
        for match in parent_matches:
            # Apply crossover and mutation to the entire array of parent indices
            new_children = self.crossover(self.pop[match])
            for child in new_children:
                children[i] = self.mutate(child)
                i += 1

        # Update the population and best individual
        self.pop = self.select_survivors(self.pop, children)
        self.pop_fit = self.f_adapt(self.pop)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)

    def run(self, ngen=10000):
        """
        Runs the evolutionary algorithm for ngen generations.

        Args:
            ngen (int): The number of generations to run the algorithm for.

        Returns:
            
            - bests (ndarray): an array with the best individual value at each 
            generation
            - means (ndarray): an array with the mean population value at each 
            generation
            - SDs (ndarray): an array with the standard deviation at each of 50 
            reference generations
        """
        # Initialize some variables for tracking progress
        means = np.zeros(ngen)
        bests = np.zeros(ngen)
        # Since it's computationally expensive, the standard deviations will 
        # only be calculated for 50 equally spaced refence generations (at most; 
        # if the algorithm converges it will be less)
        SDs = np.zeros(50)
        csum_of_means = 0
        t = 0
        t1 = time.time()
        gen_success = 0
        
        #Initialize log
        log_folder = './logs/'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log=open("logs/{}.txt".format( self.name), "w")
        
        # Print inicial
        start_msg="{}\n".format(self.name) 
        start_msg+="\nPARAMETERS={}".format(self.parameters)
        start_msg+="Iniciating...\n\n"    
        print(start_msg)
        log.write(start_msg)
        
        for it in range(ngen):
            # Update the progress counter
            progress = "Gen {}/{}".format(it, ngen)

            # Reproduce and mutate the population
            self.reproduce()

            # Calculate the mean and best adaptation of the population
            mean = np.sum(self.pop_fit) / self.n_pop
            means[it] = mean
            bests[it] = self.best_adapt

            # Add the current mean to the cumulative sum
            csum_of_means += mean

            # Clear the current line to make space for the progress update
            clear_line = "\r" + " " * len(progress) + "\r"

            # Update the progress counter
            print(progress, end=clear_line, flush=True)

            # Check for convergence every 50 generations
            if it % (ngen//50) == 0:
                # Calculate the sample DE of the means
                SDs[t] = np.sqrt(np.sum(
                    (means[:it+1] - csum_of_means/(it+1))**2) / it
                )

                # Update the progress counter
                progress = "{}: gen {}:\n".format(self.name, it)
                progress += "Best = {} Mean = {} STD = {}".format(
                    bests[it], means[it], SDs[t])
                progress += "\n"

                # Print the progress update and write to the log file
                print(progress, end="", flush=True)
                log.write(progress)

                t += 1

            # Check for convergence: in 100 generations, the best individual
            # hasn't improved (thus we have reached a plateau in exploration)
            # and the standard deviations hasn't changed (thus, the algorithm
            # has exploited that solution and the best individual has taken
            # over most of the population). 
            if it>0  \
               and abs(bests[it]-bests[it-1000]) < 0.01 \
               and (SDs[t]-SDs[t-1]) < 0:

                gen_success = it
                # Measure the time to convergence
                t2 = time.time()
                t_conver = t2 - t1

                # Print the final results
                success_msg = "{}: Convergence reached in {:.2f} seconds, after\
                            {} generations.\n".format(self.name, t_conver, ngen)
    
                success_msg += "Fitness of the best individual: \
                            {},{}".format(self.best_adapt,bests[it])
                print(success_msg, flush=True)
                log.write(success_msg + "\n")
                break
        # When the loop finishes, check if the convergence has not been reached
        if not gen_success:
            t2 = time.time()
            t_conver = t2 - t1
            # Print the final results
            success_msg = "{}: Convergence not reached in {:.2f} seconds, after\
                           {} generations.\n".format(self.name, t_conver, ngen)
            success_msg += "Fitness of the best individual: \
                            {}".format(self.best_adapt)
            print(success_msg, flush=True)
        
        #Close log
        log.close()
        
        # Plot the progress graph

        self.plot_convergence(ngen, gen_success, bests, means, SDs)

        # Todo: save best individual in a file

        return means, bests, t_conver, gen_success

    def plot_convergence(self, ngen, gen_success, bests, means, SDs):
        """
        Plots a graph showing the progress of the algorithm, with the value of 
        the best individual, the mean of the population, and the standard 
        deviation.
        Also shows the point where the convergence condition is met (if it 
        doesn't converge, this point will be at zero).

        Args:
        - ngen (int): number of max generations where the algorithm can run
        - gen_success (int): the generation number where convergence is achieved 
        (if any)
        - bests (ndarray): an array with the best individual value at each 
        generation
        - means (ndarray): an array with the mean population value at each 
        generation
        - SDs (ndarray): an array with the standard deviation at each of the 
        50 reference points

        Returns:
        - None

        """
        ngen = len(bests)
        x = np.arange(ngen)
        # Create a list with the indices of the 50 reference generations
        idx = [i for i in range(ngen) if not i % (ngen//50)]

        # Create the figure
        plt.figure(figsize=(8, 6), dpi=200)

        # Plot:
        # The mean
        plt.plot(x, means, linewidth=0.3, color="orange", label='Mean')
        # The standard deviation at the reference points
        plt.errorbar(x[idx], means[idx], yerr=SDs, color='Black',
                    elinewidth=.6, capthick=.8, capsize=.8, alpha=0.5,
                    fmt='o', markersize=.7, linewidth=.7)
        # The value of the best individual and the known optimal value
        plt.plot(x, bests[:ngen], "--", linewidth=0.8,
                color="blue", label='Best')
        if self.optimal:
            plt.plot(x, self.optimal*np.ones(ngen), label="Optimal", color="green")
        # The convergence generation on the best individual curve
        plt.scatter(gen_success, bests[gen_success],
                    color="red", label='Convergence', zorder=1)

        # Add the grid, legend and labels
        plt.grid(True)
        plt.xlim(0, ngen)
        plt.legend(loc='upper right', frameon=False, fontsize=8)
        plt.xlabel('Generations')
        plt.ylabel('Adaptation value')

        plt.title('AG {} '.format(self.name))
        #plt.show()
        # Save the figure in the plots folder and show it
        plt.savefig('plots/{}.png'.format(self.name))
