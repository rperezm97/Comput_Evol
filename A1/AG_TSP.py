import numpy as np
import random
import sys,os
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from EA import Evolutive_algorithm

class Traveling_salesman_GA(Evolutive_algorithm):
    """
    Child class implementing a genetic algorithm for the Traveling
    Salesman Problem.
    """

    def __init__(self, params, cities):
        """Initialize the algorithm withthe given parameters and cities."""
        super().__init__()
        
        self.npop = params["npop"]
        self.ps = params["ps"]
        self.t_size = params["t_size"]
        self.n_tournaments = self.npop
        self.pc = params["pc"]
        self.pm = params["pm"]

        self.cities = cities
        self.nfen = len(cities)

        self.city_distances = self.calculate_city_distances(self.cities)

    def init_pop(self):
        """Initialize the population with random permutations of the city indices."""
        city_idxs = np.arange(1, self.ncities)
        return [np.random.permutation(city_idxs) for i in range(self.npop)]

    def f_adapt(self, x):
        """Calculate the adaptation of the population based on the total distance of the routes."""
        routes = np.vstack((np.append([0], x), np.append(x, [0])))
        L = self.city_distances[tuple(routes)]
        length = np.sum(L)
        return length

    def calculate_city_distances(self, cities):
        """Calculate the pairwise distances between the cities."""
        ncities = len(cities)
        city_distances = np.empty((ncities, ncities))

        # Set 0s to diagonal elements
        city_distances[tuple((range(ncities), range(ncities)))] = 0

        # Calculate the indices of the lower triangular matrix:
        # pairs of city indices (i1, i2) where i1 < i2
        tri_idx = np.tril_indices(n=ncities, k=-1)

        # Create an array of tuples with the pairs of cities
        pairs = cities[np.array(tri_idx)]

        # Calculate the pairwise distances
        d = np.sqrt(np.sum((pairs[0]-pairs[1])**2, axis=-1))

        # Assign the distances to the lower and upper triangular parts of the matrix
        city_distances[tri_idx] = d
        permut_idx = np.roll(np.array(tri_idx), 1, axis=0)
        city_distances[tuple(permut_idx)] = d

        return city_distances

    def parent_selection(self):
        """Select parents for the next generation using tournaments."""
        parents_idx = [self.tournament_selection(self.pop_fit)
                    for _ in range(self.n_tournaments)]
        if not self.npop % 2:
            parents_idx.append(None)
        return parents_idx

    def tournament_selection(self, pop_adapt):
        """Select a parent from the population using tournament selection."""
        indiv_idx = random.sample(range(self.npop), k=self.t_size)
        idx = min(indiv_idx, key=lambda i: self.pop_adapt[i])
        return idx

    def crossover(self, parents):
        """Perform partially mapped crossover on the parents."""
        p1, p2 = parents
        if (random.random() >= self.pc 
            or not p1 or not p2):
            return p1, p2

        n = len(p1)
        # Choose two random crossover points
        b = np.random.randint(0, n-1)
        a = np.random.randint(0, n-1)
        start = min(a, b)
        end = max(a, b)

        # Create two empty children
        c1 = np.zeros(n, dtype=int)

        c2 = np.zeros(n, dtype=int)

        # Copy the segment from parent 1 to child 1
        c1[start:end+1] = p1[start:end+1]
        # Copy the segment from parent 2 to child 2
        c2[start:end+1] = p2[start:end+1]

        # Get the complement of the segment from parent 1 in child 2
        notc1 = p1-c1
        # Get the complement of the segment from parent 2 in child 1
        notc2 = p2-c2

        # Add the missing genes to child 1 and child 2
        notc1_in_notc2 = np.in1d(notc2, notc1)
        c1 += notc2*notc1_in_notc2

        notc2_in_notc1 = np.in1d(notc1, notc2)
        c2 += notc1*notc2_in_notc1

        # Find the remaining missing genes and add them to child 1 and child 2
        self.find_new_pos(p1, c1)
        self.find_new_pos(p2, c2)

        return c1, c2

    def find_new_pos(self, p, c):
        """Find the positions of the missing genes and add them to the child."""
        free = np.where(c == 0)[0]
        rest_id = np.where((1-np.in1d(p, c)))
        for i in p[rest_id]:
            pos = np.where(p == i)[0][0]
            while pos not in free:
                k = c[pos]
                pos = np.where(p == k)[0][0]
            c[pos] = i
        return c

    def mutate(self, x):
        """Perform a mutation by swapping two randomly chosen genes."""
        if random.random() >= self.pm:
            return x

        n = len(x)
        i = np.random.randint(0, n-1)
        j = np.random.randint(0, n-1)
        # Make sure i != j to introduce variation
        while j == i:
            j = np.random.randint(0, n-1)

        x[i], x[j] = x[j], x[i]
        return x

    def select_survivors(self, parents, children):
        """
        Select the next generation of individuals from the parents and children.
        """
        all_pop = np.vstack((self.pop, children))
        all_pop_fit = self.f_adapt(all_pop)
        elit_idx = np.argsort(all_pop_fit)[:self.npop]
        self.pop = all_pop[elit_idx]
        self.pop_fit = all_pop_fit[elit_idx]
        return self.pop

    def run(self, ngen):
        """Run the algorithm for the specified number of generations."""
        for i in range(ngen):
            parents_idx = self.parent_selection()
            parent_pairs = self.match_parents(parents_idx)
            children = np.empty((self.npop, self.nfen))
            for j, (p1, p2) in enumerate(parent_pairs):
                children[j*2], children[j*2+1] = self.crossover((self.pop[p1], self.pop[p2]))
            for j in range(self.npop):
                children[j] = self.mutate(children[j])
            self.pop = self.select_survivors(parents_idx, children)
            self.pop_fit = self.f_adapt(self.pop)
            self.best_adapt = np.min(self.pop_fit)
            self.best = np.argmin(self.pop_fit)
