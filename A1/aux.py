
import numpy as np


def calculate_city_distances(cities):
    """
    Calculate a matrix with the pairwise distances between the cities.
    """
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
    
def find_new_pos(p, c):
    """Find the positions of the missing genes and add them to the child."""
    free = np.where(c == 0)[0]
    rest_id = np.where((1-np.in1d(p, c)))
    for i in p[rest_id]:
        pos = np.where(p == i)[0][0]
        while pos not in free:
            k = c[pos]
            pos = np.where(p == k)[0][0]
        c[free[0]] = i
        free = free[1:]
    return c