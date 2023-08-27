import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

class Test:
    def __init__(self, EA, n_exe):
        """
        Initialize the Test class.

        Args:
        EA (Evolutive_algorithm): The evolutive algorithm object.
        n_exe (int): Number of executions for the test.
        """
        
        self.model_population = EA
        self.n_exe = n_exe

        self.VAMM = np.empty(n_exe)
        self.Mean_Error = np.empty(50)

        self.sampleVAMM = np.zeros((n_exe, self.model_population.params["ngen"]))
        self.gen_exito = np.zeros(n_exe, dtype=int)
        self.T_conver = np.empty(n_exe)

        self.TE = 0
        self.TM = 0

    def run_test(self):
        """
        Run a statistical test for evaluation of the evolutive algorithm (EA)
        with a sample of n_exe executions.
        """
        print("\nStarting parallel executions...")
        t1 = time.time()

        pool = Pool(processes=cpu_count())
        results = pool.map_async(self.run_execution, range(self.n_exe)).get()

        means, bests, t_conver, gen_success = zip(*results)
        t2 = time.time()
        self.T_exe = t2 - t1
        print("Parallel executions finished, T_exe={}".format(self.T_exe))

        self.gen_exito = np.array(gen_success)
        self.sampleVAMM = np.array(bests)
        self.T_conver = np.array(t_conver)
        self.VAMM = np.sum(self.sampleVAMM, axis=0) / self.n_exe
        print("\nVAMM: {}%".format(self.VAMM[-1]))

        ngen = self.VAMM.shape[0]
        idx = [i for i in range(ngen) if not i % (ngen // 50)]

        s_n = np.sqrt(np.sum((self.sampleVAMM[:, idx] -
                              self.VAMM[idx]
                              ) ** 2
                             , axis=0
                             ) / (self.n_exe - 1))
        # ME at 95%
        self.Mean_Error = 2 * s_n / np.sqrt(self.n_exe)

        exitos = np.sum(self.gen_exito != 0)
        self.TE = np.sum(exitos) / self.n_exe
        print("\nSuccess rate: {}%".format(100 * self.TE))
        if exitos:
            self.TM = np.sum(self.T_conver) / exitos
            print("\nAverage time: {}s".format(self.TM))

    def run_execution(self, i):
        """
        Run a single execution of the evolutive algorithm for a given index.

        Args:
        i (int): Index of the execution.

        Returns:
        tuple: mean, best, t_conver, gen_success values for the execution.
        """
        np.random.seed(i)
        self.model_population.name = "execution {}, {}".format(i, self.model_population.name)
        self.model_population.pop = self.model_population.init_pop()
        self.model_population.pop_fit = self.model_population.f_adapt(self.model_population.pop)
        self.model_population.best_adapt = np.min(self.model_population.pop_fit)
        self.model_population.best = np.argmin(self.model_population.pop_fit)
        return self.model_population.run()
    
    def plot_comparison(self, instance, experiment, ngen):
        """
        Plot the comparison of the evaluations of the genetic algorithm for a given instance.

        Args:
        instance (str): The name of the instance.
        experiment (dict): Dictionary containing the experimental results.
        ngen (int): Number of generations.
        """
        ax = plt.figure(figsize=(8, 6), dpi=200)
        for pc in experiment.keys():
            self.addplot_VAMM(ax, experiment[pc], "pc={}".format(pc))
        plt.grid(True)
        plt.xlim(0, ngen)
        plt.legend()
        plt.xlabel('Generations')
        plt.ylabel('Fitness Value')
        plt.title('Comparison of evaluations for Genetic Algorithm (instance {})'.format(instance))
        plt.grid(True)

        plt.savefig('./plots/Comparison_{}.png'.format(instance))

    def addplot_VAMM(self, ax, experiment, label):
        """
        Add the VAMM plot to the given axis.

        Args:
        ax (matplotlib.axis): The axis to plot on.
        experiment (Test): The Test object containing the experimental results.
        label (str): The label for the plot.
        """
        ngen = len(experiment.VAMM)
        x = np.arange(ngen)
        ten_idx = [i for i in range(ngen) if not i % (ngen // 50)]

        ax.plot(x, experiment.VAMM,
                linewidth=0.3, color="blue",
                label='VAMM {}. (TE={}, TM={}s)'.format(label, experiment.TE, experiment.TM))

        ax.errorbar(x[ten_idx], experiment.VAMM[ten_idx],
                    yerr=experiment.Mean_Error, color='Black',
                    elinewidth=.6, capthick=.8,
                    capsize=.8, alpha=0.5,
                    fmt='o', markersize=.7, linewidth=.7)

        plt.show()
        
        