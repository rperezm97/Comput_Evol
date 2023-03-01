import numpy as np
from ypstruct import structure
import random
from multiprocessing import Pool
from functools import partial
from pop import AE

class AG_viajante(AE):
    ""
    #%%#%%#################FUNCIONES PARA EJECUTAR EL TEST ################ #######

    def __init__(self, params, cities):
        super().__init__()
        
        self.npop = params["npop"]
        self.ps = params["ps"]
        self.t_size = params["t_size"]
        self.n_torneos = self.npop
        self.pc = params["pc"]
        self.pm = params["pm"]

        self.cities = cities
        self.nfen = len(cities)

        self.city_distances = self.city_distances(self.cities)

#%%#%%#################FUNCIONES PARA EJECUTAR EL TEST #######################

    def init_pop(self):
        city_idxs = np.arange(1, self.ncities)
        return [np.random.permutation(city_idxs)
                for i in range(self.npop)]

#%%########FUNCION DE ADAPTACION BASADA EN DISTACNCIA ################ #######

    def f_adapt(self, x):
        routes = np.vstack((np.append([0], x), np.append(x, [0])))
        L = self.city_distances[tuple(routes)]
        length = np.sum(L)
        return length

    def city_distances(self, cities):
        ncities = len(cities)
        city_distances = np.empty((ncities, ncities))

        # Asigna 0 a los elementos diagonales
        city_distances[tuple((range(ncities), range(ncities)))] = 0

        # Calcula los indices de la matriz tringular inferior:
        # de listas con el primer y el segundo indices correspondientes
        # Equivalentemente, esto son las parejas de los indices de ciudades
        # i1,i2, tales que i1<i2
        tri_idx = np.tril_indices(n=ncities, k=-1)

        # Crea un array de tuplas

        pairs = cities[np.array(tri_idx)]

        d = np.sqrt(np.sum((pairs[0]-pairs[1])**2, axis=-1))

        city_distances[tri_idx] = d

        permut_idx = np.roll(np.array(tri_idx), 1, axis=0)
        city_distances[tuple(permut_idx)] = d

        return city_distances


#%%#%%#############FUNCIONES PARA SELECCION POR TORNEO ################ #######

    def select(self, pop, pop_adapt):
        parents_idx = [self.torneo(pop_adapt)
                       for _ in range(self.n_torneos)]
        if not self.npop % 2:
            parents_idx.append(None)
        return parents_idx

    def torneo(self, pop_adapt):
        indiv_idx = random.sample(range(self.npop), k=self.t_size)
        idx = min(indiv_idx, key=lambda i: self.pop_adapt[i])
        return idx


#%%#%%##############FUNCIONES PARA CRUCE PARCIAL MAPEADO ################ #######

    def cruce(self, p1, p2, pc):
        if (random.random() >= pc 
            or not p1 or not p2):
            return p1, p2
        
        
        n = len(p1)
        # Hacemos qeu sea random
        b = np.random.randint(0, n-1)
        a = np.random.randint(0, n-1)
        so = min(a, b)
        sf = max(a, b)
        so = 2
        sf = 6
        c1 = np.zeros(n, dtype=int)
        c2 = np.zeros(n, dtype=int)
        # Copiamos el segmento seleccionado del padre p1=<12345678> en el hijo
        # correspondiente c1=<**3456**>
        c1[so:sf+1] = p1[so:sf+1]
        c2[so:sf+1] = p2[so:sf+1]
        # Cogemos el complemento del segmento notc1=
        notc1 = p1-c1
        notc2 = p2-c2
        # Del resto de los genes <12****78>, queremos añadir los no están en el
        # segmento del otro padre c1 (o lo que es lo mismo, cuales sí están en
        # el complemento not2) al hijo en la posición del otro padre.
        # La función np.in1d(notc2, notc1) nos devuelve un 1 en la po
        # un array booleano con un 1 en la posición de los genes seleccionados.
        # Añadimos los genes seleccionados en la posición
        n1 = np.in1d(notc2, notc1)
        n2 = np.in1d(notc1, notc2)

        c1 += notc2*n1
        c2 += notc1*n2

        # Cogemos el resto, que son aquellos en los que todavía hay un hueco en
        # el otro hijo
        self.find_new_pos(p1, c1)
        self.find_new_pos(p2, c2)

        return c1, c2

    def find_new_pos(self, p, c):
        free = np.where(c == 0)[0]
        rest_id = np.where((1-np.in1d(p, c)))
        for i in p[rest_id]:
            pos = np.where(p == i)[0][0]
            while pos not in free:
                k = c[pos]
                pos = np.where(p == k)[0][0]
            c[pos] = i
        return c

    def match_padres(self, padres_idx):
        # Haz parejas de padres. Si son impares se ignora el último
        # (esto se tiene en cuenta en la función de selección de supervivientes)
        middle = (self.npop-self.npop % 2)//2
        matches = zip(padres_idx[:middle], padres_idx[middle:])
        return matches

#%%#%%##############FUNCIONES PARA MUTACION POR INTERCAMBIO ################ #######

    def mutate(self, x):
        if random.random() >= self.pm:
            return x

        n = len(x)
        i = np.random.randint(0, n-1)
        j = np.random.randint(0, n-1)
        # Nos aseguramos que i=!j, para introducir variación
        while j == i:
            j = np.random.randint(0, n-1)

        x[i], x[j] = x[j], x[i]
        return x


#%%#%%##############FUNCIONES PARA SELECCION SUPERVIVIENTES ################ #######


    def select_superv(self, pop, pop_adapts, hijos):
        new_pop = np.empty(self.pop.shape)

        for idx in elit_idx:
            best_id = np.argmin(pop_adapt)
