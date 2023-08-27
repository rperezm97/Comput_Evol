import numpy as np
from abc import ABC
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class Evolutive_algorithm(ABC):
    """Abstract class that implements the methods of a generic evolutionary
    algorithm and serves as a parent for other algorithms."""

    def __init__(self, instance_id, base_folder=None, optimal=None,
                 convergence_thresholds=(1e-5,.3,.1)):
        """Initialize the population, calculate the adaptation of each
        individual, and store the best individual and its adaptation."""
        self.instance_id = instance_id
        self.init_parameters()
       
        
        self.base_folder=base_folder
        
        self.pop = self.init_pop()
        self.pop_fit = self.f_fit(self.pop)
        
        self.best = np.argmin(self.pop_fit) 
        self.best_adapt = self.pop_fit[self.best]
        self.optimal=optimal
        
        self.convergence_thresholds=convergence_thresholds
        
        
    def init_pop(self):
        """Initialize the population. To be implemented by child classes."""
        pass
    
    def init_parameters(self):
        pass 
    
    def f_fit(self, pop):
        """Calculate the fitness value of each individual in the population.
        To be implemented by child classes."""
        pass

    def parent_selection(self):
        """Select parents from the population. It should return a list with
        the indices of the parents. To be implemented by child 
        classes."""
        pass

    def variation_operators(self, parents):
        """Perform crossover and mutation on two parents to create two children. To be
        implemented by child classes."""
        pass

    def select_survivors(self, children):
        """Select the individuals to be included in the next generation.
        To be implemented by child classes."""
        pass

        
    def convergence_condition(self,it, bests, SDs, means):
        
        change_threshold,sd_threshold,optimal_threshold=self.convergence_thresholds
        
        l_interval = self.n_gen // 100
        t = it // l_interval
        
        sd_rate=(SDs[t] / (np.mean(SDs[:t])))
        change_rate=((bests[it-l_interval] - bests[it])/(bests[it-l_interval]))
        condition_diversity=(change_rate< change_threshold 
                            and sd_rate< sd_threshold)
        
        print("Rate of best_change:{}\nRate of SD change:{}\n".format(
              change_rate,sd_rate))
        
        
        if not (self.optimal is None):
            if self.optimal!=0:
                percentage_optimal = (bests[it] / self.optimal)-1
            else: 
                percentage_optimal=np.abs(bests[it]) 
            condition_optimal = percentage_optimal < optimal_threshold        

            print("Distance from bests towards know optimal: {}%".format(int(percentage_optimal*100)))
            return condition_optimal or condition_diversity
        else:
            return condition_diversity
   
    def check_success(self,bests,gen_convergence,t_converg):
        return gen_convergence>0
    
    def reproduce(self):
        """Perform reproduction for 1 generations."""
        # Select parents and create the matches by reshape the parent indices
        # array into a 3D array (for indexing the population matrix)
        parent_matches = self.parent_selection()

        
        children=self.variation_operators(parent_matches)

        # Update the population and best individual
        self.select_survivors(children)
        
    def run(self, stop_after_convergence=True):
        """
        Runs the evolutionary algorithm for ngen generations.

        Args:
            ngen (int): The number of generations to run the algorithm for.

        Returns:

            - bests (ndarray): an array with the best individual value at each 
            generation
            - means (ndarray): an array with the mean population value at each 
            generation
            - SDs (ndarray): an array with the standard deviation at each of 100 
            reference generations
        """
        # Initialize some variables for tracking progress
        means = np.zeros(self.n_gen)
        bests = np.zeros(self.n_gen)
        # Since it's computationally expensive, the standard deviations will
        # only be calculated for 100 equally spaced refence generations (at most;
        # if the algorithm converges it will be less)
        SDs = np.zeros(100)
        t0 = time.time()
        gen_conver = 0

        # Initialize log
        log_folder=os.path.join(self.base_folder,"logs")
        if not os.path.exists(log_folder):
            os.makedirs(self.log_folder)
        log_file_name="{}.txt".format(self.instance_id)
        log = open(os.path.join(log_folder,log_file_name),
                   "w")

        # Print inicial
        start_msg ="Running {} for {} generations:\n".format(self.instance_id,self.n_gen)
        print(start_msg)

        for it in tqdm(range(self.n_gen)):
            # Reproduce and mutate the population
            self.reproduce()

            # Calculate the mean and best adaptation of the population

            mean = np.sum(self.pop_fit) / self.n_pop
            means[it] = mean
            bests[it] = self.best_adapt
            
            # Check for convergence every 100 generations
            if not it % (self.n_gen//100):
                t=it//(self.n_gen//100)
                # Calculate the sample DE of the means
                SDs[t] = np.sqrt(np.sum(
                    (self.pop_fit - mean)**2) /  (self.n_pop-1)
                )

                # Update the progress counter
                progress = "\nGen {}:\n".format(it)
                progress += "Best = {} \nMean = {} \nSTD = {}\n ".format(
                    bests[it], means[it], SDs[t])
                progress += "\n"

                # Print the progress update and write to the log file
                print(progress, end="", flush=True)
                log.write(" ".join(map(str,
                                   [it,bests[it], means[it], SDs[t], "\n"])))

                # Check for convergence: in 100 generations, the best individual
                # hasn't improved (thus we have reached a plateau in exploration)
                # and the standard deviations hasn't changed (thus, the algorithm
                # has exploited that solution and t,e best individual has taken
                # over most of the population).
                if (t>0 and 
                    self.convergence_condition(it,
                                               bests=bests,
                                               SDs=SDs,
                                               means=means)
                    and not gen_conver):

                    gen_conver = it
                    # Measure the time to convergence
                    t1 = time.time()
                    t_conver = t1 - t0

                    # Print the final results
                    success_msg = "\n Convergence reached in {} seconds, after {} generations.\n".format(t_conver,
                                                        gen_conver)

                    success_msg += "Fitness of the best individual: {}".format(self.best_adapt)
                    print(success_msg, flush=True)
                    if stop_after_convergence:
                        break

        # Set the best and mean values of the generations after convergence as
        # the value of convergence, for the plots
        bests[it:] = bests[it]
        means[it:] = means[it]
        # When the loop finishes, check if the convergence has not been reached
        if not gen_conver:
            t1 = time.time()
            t_conver = t1 - t0
            gen_converg=0
            # Print the final results
            success_msg = "{} - Convergence not reached in {:.2f} seconds, after {} generations.\n".format(self.instance_id, t_conver,
                                                     self.n_gen)
            success_msg += "Fitness of the best individual: {}".format(self.best_adapt)
            print(success_msg, flush=True)

        log.write(" ".join(map(str,
                           [gen_conver, 
                            t_conver,
                            "\n"])))
        log.write(str(self.pop[self.best]))
        # Close log
        log.close()

        # Plot the progress graph

        self.plot_convergence(gen_conver, bests, means, SDs)

        # Todo: save best individual in a file

        return means, bests, t_conver, gen_conver
    
    
    def plot_convergence(self, gen_converg, bests, means, SDs):
        """
        Plots a graph showing the progress of the algorithm, with the value of 
        the best individual, the mean of the population, and the standard 
        deviation.
        Also shows the point where the convergence condition is met (if it 
        doesn't converge, this point will be at zero).

        Args:
        - ngen (int): number of max generations where the algorithm can run
        - gen_conver (int): the generation number where convergence is achieved 
        (if any)
        - bests (ndarray): an array with the best individual value at each 
        generation
        - means (ndarray): an array with the mean population value at each 
        generation
        - SDs (ndarray): an array with the standard deviation at each of the 
        100 reference points

        Returns:
        - None

        """
        n_gen_exe= gen_converg if gen_converg>0 else self.n_gen
        generations = np.arange(n_gen_exe)
        # Create a list with the indices of the 100 reference generations
        ref_generations = [i for i in range(n_gen_exe) if not i % (self.n_gen//100)]
        # Create the figure
        plt.figure(figsize=(8, 6), dpi=200)
        plt.xlim(0,n_gen_exe)
        plt.ylim(self.optimal-0.05*max(bests),1.05*max(bests))

        # Plot:
        # The mean
        
        plt.plot(generations,
                 means[:n_gen_exe], 
                 linewidth= .3,
                 color="orange",
                 label='Mean')
        # The standard deviation at the reference points
        plt.errorbar(ref_generations, 
                     means[ref_generations], 
                     yerr=SDs[:100*n_gen_exe//self.n_gen], 
                     color='Black',
                     elinewidth=.6,
                     capthick=.8, 
                     capsize=.8, 
                     alpha= .5,
                     fmt='o',
                     markersize=.7,
                     linewidth=.7)
        # The value of the best individual and the known optimal value
        plt.plot(generations,
                 bests[:n_gen_exe],
                 "--", 
                 linewidth= .8,
                 color="blue",
                 label='Best')
        
        if not (self.optimal is None):
            plt.plot(generations, 
                     self.optimal*np.ones(n_gen_exe),
                     label="Optimal", 
                     color="green")
        # The convergence generation on the best individual curve
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
        
        #plt.yscale('log')
        # Create the axis
        plt.grid(True)
        
        # Show the plot
        plt.show()

        # Save the figure in the plots folder and show it
        plt.savefig(os.path.join(self.base_folder,
                            'plots/{}.png'.format(self.instance_id)))