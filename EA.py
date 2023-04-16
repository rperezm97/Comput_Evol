import numpy as np
from abc import ABC

class Evolutive_algorithm(ABC):
    """Abstract class that implements the methods of a generic evolutionary
    algorithm and serves as a parent for other algorithms."""

    def __init__(self):
        """Initialize the population, calculate the adaptation of each
        individual, and store the best individual and its adaptation."""
        self.pop = self.init_pop()
        self.pop_fit = self.f_adapt(self.pop)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)
        

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
        i=0
        for match in parent_matches:
            # Apply crossover and mutation to the entire array of parent indices
            new_children = self.crossover(self.pop[match])
            for child in new_children:
                children[i] = self.mutate(child)
                i+=1

        # Update the population and best individual
        self.pop = self.select_survivors(self.pop, children)
        self.pop_fit = self.f_adapt(self.pop)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)
    def run(self,ngen):
        for it in range(ngen):
            # Para cada generación, imprime el contador
            print("Gen {}/{}".format(it, ngen), 
                    end='\r')

            # Reproduce y muta la población
            self..reproduce()

            # Guarda la media de adaptaciones de la población y el
            # valor del mejor individuo en los array correspondiente
            mean = np.sum(self.adapts)/population.npop
            means[it] = mean
            bests[it] = population.bestadapt

            # Añade la ultima media a la suma cumulativa
            csum_of_means += mean
            
            #Actualiza el contador
            print(end=LINE_CLEAR)
            
            # Calcula la DE, muestra el progreso y comprueba la condición
            # de convergencia para 50 generaciones de las totales:
            if not it % (ngen//50) and it > 0:
                # Calculamos y guardamos la DE muestral de las medias:
                # Nótese que como it empieza en 0, el numero de 
                # generaciones total hasta el momento es it+1. La media
                # de las medias viene dada por csum_of_means/(it+1).
                DEs[t] = np.sqrt(np.sum(  
                                    (means[:it+1]-
                                        csum_of_means/(it+1)
                                        )**2
                                        )/it)
                                    
                # Muestra el progreso
                print("Exe {}: Generación {}:\n".format(i,it),
                        "Best = {} Mean = {} STD = {}".format(
                                                        bests[it],
                                                        means[it], 
                                                        DEs[t]))
                print("Exe {}: Generación {}:\n".format(i,it),
                        "Best = {} Mean = {} STD = {}".format(
                                                        bests[it],
                                                        means[it], 
                                                        DEs[t]),
                        file=log)
                t+=1
                
            #Condición de convergencia
            if(not gen_exito 
                and abs(bests[it]-bests[it-100]) < 0.01
                # Añadimos esta condición porque a veces no cambia en 
                # 100 generaciones pero la DE sigue creciendo. Esta 
                # segudna condición nos da mayor confianza en la
                # convergencia
                and (DEs[t]-DEs[t-1]) < 0
            ):
                gen_exito = it
                print("----EXE {} CONVERGE (EN gen={})---".format(i,
                                                                    it)
                        ,flush=True)
                print("----EXE {} CONVERGE (EN gen={})---".format(i,
                                                                    it)
                        ,file=log)     
                
                # Ahora, mide el tiempo, 
                t2 = time.time()
                t_conver=t2-t1
                print("\nTiempo de convergencia={}s".format(t_conver),
                        flush=True)
                print("\nTiempo de convergencia={}s".format(t_conver), 
                        file=log)
                
                if stop:
                    break