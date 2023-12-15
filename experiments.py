import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from multiprocessing import Pool, cpu_count

class Experiment:
    def __init__(self, EA, n_exe, initial_exe):
        """
        Initialize the Experiment class.

        Args:
        EA (Evolutive_algorithm): The evolutive algorithm object.
        n_exe (int): Number of executions for the experiment.
        """

        print("\n\n -------- STARTING STATISTICAL TESTS ------")
        self.model_AE = EA
        self.instance_id = self.model_AE.instance_id
        self.n_exe = n_exe
        self.initial_exe = initial_exe

        self.MBF = np.empty(100)
        self.ME_MBF = np.empty(100)

        self.sample_bests = np.zeros((n_exe, 100))
        self.sample_means = np.zeros((n_exe, 100))
        self.sample_DEs = np.zeros((n_exe, 100))
        self.sample_gen_converg = np.zeros(n_exe, dtype=int)

        self.sample_best = np.zeros((n_exe, 100))
        self.sample_T_exe = np.empty(n_exe)

        self.successes = np.empty(n_exe, dtype=bool)
        self.SR = 0

        self.MT = 0
        self.ME_MT = 0
    
    # def check_success(self,checkgen_i, bests):

    #     if not (self.optimal is None):
    #         distance_optimal = np.abs(bests[checkgen_i] - self.optimal)-1
    #         condition_optimal = distance_optimal < self.optimal_threshold

    #         print("Distance from bests towards know optimal: {}%".format(
    #             int(distance_optimal*100)))
    #         return condition_optimal
    #     #return gen_convergence>0
    def run_experiment(self,):
        """
        Run a statistical experiment for evaluation of the evolutive algorithm (EA)
        with a sample of n_exe executions.
        """
        print("\nStarting parallel executions...")
        t1 = time.time()
        
        pool = Pool(processes=cpu_count())
        a=pool.map_async(self.run_execution,
                         list(range(self.initial_exe, self.n_exe))).get()
        
        t2 = time.time()
        t_exe = t2 - t1

        print(f"""Executions {self.initial_exe} to {self.n_exe} finished, 
              time eluted= {t_exe}""")
        #self.estimate_indices()

    def run_execution(self,execution_i):
        np.random.seed(execution_i)

        self.model_AE.instance_id = self.instance_id+f"_EXE{execution_i}"
        self.model_AE.pop = self.model_AE.init_pop()
        self.model_AE.pop_fit = self.model_AE.f_fitness(self.model_AE.pop)
        self.model_AE.best_adapt = np.min(self.model_AE.pop_fit)
        self.model_AE.best = np.argmin(self.model_AE.pop_fit)
        print(f"----------------EXECUTION {execution_i}---------------")
        return self.model_AE.evolve()
        
    def estimate_indices(self):
        log_folder = os.path.join(self.model_AE.base_folder, "logs")
        execution_logs = [log_file for log_file in os.listdir(log_folder)
                          if ((log_file[:len(self.instance_id)] == self.instance_id)
                              and int(log_file[-5]) < self.n_exe)]
        if len(execution_logs) < self.n_exe:
            print("Some execution is missing, check the folder")
            return

        for exe_i, log_file in enumerate(execution_logs):
            print(log_file[-8:-4])
            if log_file[-8:-4] != f"EXE{exe_i}":
                continue
            with open(os.path.join(log_folder, log_file)) as fp:
                f = fp.readlines()
                for i_rec, line in enumerate(f):
                    if line[0] == "[":
                        break
                i_rec_converg = i_rec-1

                gen_converg, t_converg, _ = f[-1].split(" ")
                gen_converg = int(gen_converg) or 0
                t_conver = float(t_converg)

                (_,
                 self.sample_bests[exe_i, :i_rec_converg],
                 self.sample_means[exe_i, :i_rec_converg],
                 self.sample_DEs[exe_i, :i_rec_converg]
                 ) = np.loadtxt(f[:i_rec_converg], dtype=np.float32).T

            self.sample_gen_converg[exe_i] = gen_converg

            n_gen_exe = gen_converg if gen_converg > 0 else self.model_AE.n_gen
            t = 100*n_gen_exe//self.model_AE.n_gen
            self.sample_bests[exe_i,
                              i_rec_converg:] = self.sample_bests[exe_i, i_rec_converg-1]
            self.sample_bests[exe_i,
                              i_rec_converg:] = self.sample_bests[exe_i, i_rec_converg-1]
            self.sample_T_exe[exe_i] = t_conver
            self.successes[exe_i] = self.model_AE.check_success(self.sample_bests[exe_i, :],
                                                                gen_converg,
                                                                t_converg)

        self.MBF, self.ME_MBF = self.confidence_interval(
            sample=self.sample_bests)
        print(f"\nMBF: {self.MBF[-1]} +- {self.ME_MBF[-1]}")

        self.SR = np.sum(self.successes) / self.n_exe
        print(f"\nSuccess rate: {100 * self.SR}%")

        sample_T_exe_success = self.sample_T_exe[self.successes]
        if len(sample_T_exe_success):

            self.MT, self.ME_MT = self.confidence_interval(
                sample_T_exe_success)

        print(f"\nAverage time: {self.MT} +- {self.ME_MT}s")
        self.plot_MBF()
    
    def confidence_interval(self, sample):
        n_sample = sample.shape[0]
        mean = np.sum(sample, axis=0) / n_sample
        # Confidence_interval
        quasisd = np.sqrt(np.sum((sample - mean) ** 2, axis=0
                               ) / (n_sample - 1))
      

        # Nivel de confianza deseado (por ejemplo, 95%)
        confidence_level = 0.95

        # Grados de libertad (relacionados con el tamaño de las muestras)
        degrees_of_freedom = n_sample-1  # Cambia esto al número adecuado de grados de libertad

        # Calcular el valor crítico t
        critical_value_t = stats.t.ppf((1 + confidence_level) / 2, 
                                       degrees_of_freedom)

        ME = critical_value_t * quasisd / np.sqrt(n_sample)
        return mean, ME

    def plot_MBF(self, ax=None):
        """
        Add the MBF plot to the given axis.

        Args:
        ax (matplotlib.axis): The axis to plot on.
        experiment (Experiment): The Experiment object containing the experimental results.
        label (str): The label for the plot.
        """

        ngen = len(self.MBF)

        adding_plot = ax
        if not adding_plot:
            ax = plt.figure(figsize=(8, 6), dpi=200)
            plt.grid(True)
            plt.xlim(0, ngen)
            plt.xlabel('Generations')
            plt.ylabel('Fitness Value')
            plt.title(f"""Statistical estimation for {self.instance_id}.
                      \n Sample size={self.n_exe}""")
            plt.grid(True)

        x = np.arange(ngen)

        plt.plot(x, self.MBF,
                 linewidth=1,
                 label=f"MBF of {self.instance_id} (SR={self.SR}, MT={int(self.MT)}+-{int(self.ME_MT)}s)")

        plt.errorbar(x, self.MBF,
                     yerr=self.ME_MBF, color='Black',
                     elinewidth=.6, capthick=.8,
                     capsize=.8, alpha=0.5,
                     fmt='o', markersize=.7, linewidth=.7)
        if adding_plot:
            return ax
        else:
            plt.legend()
            plt.show()


if __name__ == "__main__":
    import sys
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    print(current)
    sys.path.append(current)
    sys.path.append(parent)
    from A1_Genetic_algorithm.GA import Genetic_Algorithm
    # ga=Genetic_Algorithm_TSP("complex_02")
    # t=Experiment(ga, n_exe=5, initial_exe=2,t_n=2)
    # t.run_experiment()
    # t.plot_MBF()
    ga = Genetic_Algorithm("Sphere_B1")
    t = Experiment(ga, n_exe=2, initial_exe=0)
    t.run_experiment()
