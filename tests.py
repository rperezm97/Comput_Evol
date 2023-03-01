#%%#################### IMPORTS Y VARIABLES GLOBALES ##########################
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from pop import Population

# Usamos este string para tener un contador de generaciones.
# (Con print(end=LINE_CLEAR), se borra la ultima línea impresa en la consola).
LINE_CLEAR = '\x1b[2K'

###############################################################################
# %%           CLASE DE TESTS PARA EL ALGORITMO GENÉTICO                      #
###############################################################################

class Test:
    def __init__(self, name, instance, n_exe=30, output=False):
        """ 
        instance:string. Toma el valor "simple" para la instancia simple (101  
        ciudades o "complex" para la instancia compleja (100010 ciudades).

        n_exe: número de ejecuciones independientes del algoritmo para el test,
        esto es, tamaño de la muestra.

        param_file:string o None. Es la ubicación del archivo .txt con los
        valores de los parámetros del algoritmo, o None para cargar los valores
        por defecto. Ver función load_params abajo.
        
        output: Bool. Es la ubicación del archivo donde guardar 
        la mejor ruta encontrada. con los valores de  
        los parámetros del algoritmo, o None para cargar los valores por 
        defecto. Ver función load_params abajo.
        """
        # Inicializa los parametros de la clase.
        self.name=name
        self.instance = instance
        self.n_exe = n_exe
        self.output=output
        
        # Carga los parámetros, las ciudades y el óptimo conocido
        self.param_file='./params/{}/{}'.format(
                                            self.instance, 
                                            name)+".json"
                                    #todo os join
        self.params = self.load_params()
        self.cities, self.optimal = self.load_instance()
        
        print("Inicializando población, calculando matriz de distancias...")
        self.model_population = Population(self.params, 
                                           self.cities)
        print("Done")
        
        self.VAMM=np.empty(n_exe)
        self.ME=np.empty(50)
        
        self.sampleVAMM= np.zeros((n_exe, self.params["ngen"]))
        self.gen_exito=np.zeros(n_exe,dtype=int)
        self.T_conver=np.empty(n_exe)
        
        self.TE=0
        self.TM=0
        if output:
            self.best_routes= np.empty((n_exe, 
                                        (len(self.cities)-1)),
                                       dtype=int)
        else:
            self.best_routes= np.empty(n_exe,
                                        dtype=int)
        self.initial=False

     

#%%#%%#################FUNCIONES PARA EJECUTAR EL TEST ################ #######
   
    def run(self):
        print("\n Inicando ejecuciones en paralelo...")
        t1=time.time()
        start=int(self.initial) 
        pool = Pool(processes=cpu_count())
        #Si se ha hecho la primera evalaución incial, entonces ya tenemos el 
        #valor de la ejecución 0, en cuyo caso nos quedan n_exe-1 ejecuciones
        r = pool.map_async(self.execute, 
                           range(start,self.n_exe)
                           ).get()
        gen_exito, bests, best_route,t_conver=zip(*r)
        t2=time.time()
        self.T_exe=t2-t1
        print(" Ejecuciones en paralelo finalizadas, T_exe={}".format(self.T_exe))
        #Usa r.get() para obtener los valores de las ejecuciones en paralelo,
        #y convierte el resultado a un array para poder hacer unpacking en 
        #los arrays corresponientes. 
        
        self.gen_exito[start:]=gen_exito
        self.sampleVAMM[start:]=bests
        self.best_routes[start:]=best_route
        self.T_conver[start:]=t_conver
        self.VAMM = np.sum(self.sampleVAMM, axis=0)/self.n_exe
        print("\nVAMM:{}%".format(self.VAMM[-1]))
        ngen=self.VAMM.shape[0]
        idx=[i for i in range(ngen) if not i % (ngen//50)]

        s_n = np.sqrt(np.sum((self.sampleVAMM[:,idx]-
                              self.VAMM[idx]
                              )**2
                             ,axis=0
                             )/(self.n_exe-1))
        #ME al 95%
        self.ME = 2*s_n/np.sqrt(self.n_exe)
        
        
        exitos=np.sum(self.gen_exito!=0)
        self.TE=np.sum(exitos)/self.n_exe
        print("\nPorcentaje de éxito:{}%".format(100*self.TE))
        if exitos:
            self.TM=np.sum(self.T_conver)/exitos
            print("\nTiempo medio :{}s".format(self.TM))
    
    
    def initial_evaluation(self):
        #La ejecución del algoritmo nos devuelve

        gen_exito, bests, best_route,t_conver =self.execute(
                                                            i=0, 
                                                            stop=False)
        self.gen_exito[0]=gen_exito
        self.sampleVAMM[0]=bests
        self.best_routes[0]=best_route
        self.T_conver[0]=t_conver
        
        self.initial=True

    def execute(self, i, stop=True):
        """
        Hace una ejecución inicial con una sola población y genera un gráfico 
        de convergencia con el que se puede estimar visualmente el numero de
        generaciones necesaria para convergencia. 
        Como referencia, devuelve y plotea la generación a partir de la cual se 
        cumple la condición de convergencia: que el valor del mejor individuo 
        no cambie en 100 generaciones (como recomedado en el texto base) y que 
        la desviación estandar de la media se reduzca.También guarda un log con
        el mejor individuo, la media y la DE de 50 generaciones.
        """
        test_execution=(i==1)
        # Guardamos el tiempo inicial para medir el tiempo de ejecución.
        t1 = time.time()

        ##### INICIALIZACIÓN DE PARAMETROS #####
        # Inicializamos la población, el numero de generaciones máxima
        # y los arrays para el mejor individuo, la media y la DE, y
        # la generación de convergencia (0 por defecto)
        population=self.model_population
        # Re-inicializamos la población modelo (las ejecuciones del
        # algortimo en paralelocorre con copias de la misma). Sin embargo, 
        # mantenemos city_distances (esto acelerará los cálculos)
        population.reset()
        
        ngen = self.params["ngen"]
        
        bests = np.zeros(ngen)
        means = np.zeros(ngen)
        DEs = np.zeros(50)
        
        gen_exito = 0
        t_conver=0
        
        # Inicializa una suma cumulativa de medias, que nos permitirá calcular
        # la DE de las medias de forma más eficiente que con np.std()
        csum_of_means = 0
        t=1
        # Sobreescribe el log anterior. Si no existe lo crea.
        with open("./logs/{}/Exe{}_{}.txt".format( self.instance, 
                                                    i, 
                                                    self.name),
                  "w") as log:
           
            # Print inicial
            print("EJECUCIÓN {}\n".format(i), 
                  "INSTANCIA={} \nPARAMETROS={}".format( 
                                                    self.instance,  
                                                    self.params),
                  "Iniciando...\n\n")
            print("EJECUCIÓN {}\n".format(i), 
                  "INSTANCIA={} \nPARAMETROS={}".format( 
                                                    self.instance,  
                                                    self.params),
                  "Iniciando...\n\n",
                  file=log)
            ######### BUCLE PRINCIPAL #########
            for it in range(ngen):
                    # Para cada generación, imprime el contador
                    print("Generación {}/{}".format(it, ngen), 
                          end='\r')
        
                    # Reproduce y muta la población
                    population.reproduce()
        
                    # Guarda la media de adaptaciones de la población y el
                    # valor del mejor individuo en los array correspondiente
                    mean = np.sum(population.adapts)/population.npop
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
            #FIN DEL BUCLE.    
                           
            bests[it:]= bests[it]
            means[it:] =means[it]
            
            if not gen_exito:
                print("---EXE {} NO CONVERGE (EN {} GENERACIONES)---".format(i,
                                                                     it),
                      flush=test_execution)
                print("\nTiempo de convergencia={}s".format(t_conver), 
                      file=log)
                
                
            print("\nCOSTE MEJOR RUTA ={}:".format(
                                                population.bestadapt),
                   file=log)
            #guarda la generación de
            # convergencia y plotea la grafica de progreso
            
            self.plot_converg(i,
                              gen_exito,
                              bests,
                              means, 
                              DEs)
            self.city_distances=population.city_distances
            
            if self.output: 
                best_route=population.pop[population.best]
                #Todo: guardar
            else:
                best_route=0
        return gen_exito, bests, best_route, t_conver

    def plot_converg(self, i, gen_exito, bests, means, DEs,):
        """
        Plotea el gráfico de progreso, con el valro del mejor individuo, de la
        media de la población y de la desviación estandard. También muestra el
        punto donde se cumple al condición de convergencia (si no converge,
        este punto estará en el cero)
        """
        # Creo el eje x desde 0 hasta ngen
        ngen = self.params["ngen"]
        x = np.arange(ngen)
        # Creo al lista de indices de las 50 generaciones de referencia
        idx = [i for i in range(ngen) if not i % (ngen//50)]

        # Creo la figura
        plt.figure(figsize=(8, 6),dpi=200)

        # PLOTEO:
        # La media
        plt.plot(x,means, linewidth=0.3,
                 color="orange", label='Media')
        # Las desviacion tipica en los 50 puntos
        plt.errorbar(x[idx],means[idx], 
                     yerr=DEs, color='Black', 
                     elinewidth=.6, capthick=.8, 
                     capsize=.8, alpha=0.5, 
                     fmt='o', markersize=.7, linewidth=.7)
        # El valor del mejor individuo y el óptimo conocido
        plt.plot(x, bests[:ngen], "--", linewidth=0.8,
                 color="blue", label='Mejor')
        plt.plot(x, self.optimal*np.ones(ngen), 
                 label="Óptimo", color="green")
        # La generación de convergencia sobre la curva del mejor individuo
        plt.scatter(gen_exito, bests[gen_exito],
                    color="red", label='Convergencia', zorder=1)

        # Añado la cuadricula, la leyenda y las etiquetas
        plt.grid(True)
        plt.xlim(0, ngen)
        plt.legend(loc='upper right', frameon=False, fontsize=8)
        plt.xlabel('Generaciones')
        plt.ylabel('Valor de adaptación')

        plt.title('AG {} Instancia={}, - Exe {}'.format(
                                                        self.name,
                                                        self.instance,  
                                                        i))
        # Guardo la figura en la carpeta plots y muetsro la figura
        plt.savefig('./plots/{}/Exe{}_{}.png'.format(
                                                    self.instance, 
                                                    i, 
                                                    self.name))

    def addplot_VAMM(self, label):
        
        ngen = len(self.VAMM)
        x = np.arange(ngen)
        ten_idx = [i for i in range(ngen) if not i % (ngen//50)]
        
        plt.plot(x, self.VAMM,
                linewidth=0.3,color="blue", 
                label='VAMM {}. (TE={}, TM={}s)'.format(
                                                    self.name,
                                                    self.TE,
                                                    self.TM))

        plt.errorbar(x[ten_idx], self.VAMM[ten_idx], 
                    yerr=self.ME, color='Black', 
                    elinewidth=.6, capthick=.8, 
                    capsize=.8, alpha=0.5, 
                    fmt='o', markersize=.7, linewidth=.7)
        
        
        #plt.show()
        
    # ##### FUNCIONES AUXILIARES #####:
   
    # ##### FUNCIONES PRINCIPALES #####:

    # def execute(self, gen_conver=None):
    #     """
    #     Plotea el gráfico de progreso, con el valro del mejor individuo, de la media 
    #     de la población y de la desviación estandard. También muestra el punto donde 
    #     se cumple al condición de convergencia (si no ah convergido, este punto estará en
    #     el cero)
    #     """
        
        
    #     # Guardamos el tiempo inicial para medir el tiempo de ejecución.
    #     t1 = time.time()

    #     ##### INICIALIZACIÓN DE PARAMETROS #####
    #     # Crea un diccionario de poblaciones de tamaño n_exe,
    #     # cuyos valores de adaptación constituyen la muestra aleatoria
    #     # con la que evaluaremos el algoritmo
    #     print("Inicializando muestra...")
    #     self.sample = {i: Population(self.params, self.cities, self.city_distances)
    #                    for i in range(self.n_exe)}
    #     self.exito=np.zeros(self.n_exe, dtype=bool)
    #     self.T_conver=np.zeros(self.n_exe, dtype=bool)
    #     if gen_conver:            
    #         ngen=gen_conver
    #     else:
    #         ngen=self.params["ngen"]
    #     # npop=self.params["npop"]
    #     self.VAMM = np.zeros(ngen)
    #     # self.gen_mean=np.zeros(ngen)
    #     self.ME = np.zeros(50)
    #     # Inicializamos una suma cumulativa de medias, que nos permitirá calcular
    #     # la desviación estandar de las medias de forma más eficiente que con np.std()

    #     prev_best= np.zeros((self.n_exe,100))
    #     t=0
    #     # # Sobreescribimos el log anterior con el mismo nombre. Si no existe lo crea.
    #     with  open(".\logs\log_execution_{}_pc{}.txt".format(
    #             self.instance, self.params["pc"]), "w") as log:
           
    #         # Print inicial
              
    #         print("EVALUACIÓN DE ALGORITMO GENÉTICO \n",
    #               "INSTANCIA={} ,EJECUCIONES={}\nPARAMETROS={} \nIniciando...\n\n ".format(
    #                   self.instance, self.n_exe, self.params))
    #         print("EVALUACIÓN DE ALGORITMO GENÉTICO \n",
    #               "INSTANCIA={} ,EJECUCIONES={}\nPARAMETROS={} \nIniciando...\n\n ".format(
    #                   self.instance, self.n_exe, self.params), file=log)
    
    #         ######### BUCLE PRINCIPAL #########

    #         for it in range(ngen):
               
    #             VAMM = np.sum(best)/self.n_exe
    #             self.VAMM[it] = VAMM
    #             #Actualiza el contador
    #             print(end=LINE_CLEAR)
                
    #             # Calculamos la media y la añadimos al array correspondiente
    #             # mean=[np.sum(self.sample[i].adapts)/npop for i in range(self.n_exe)]
    #             # self.gen_mean[it] = np.sum(mean)/self.n_exe
    #             if not it % (ngen//50):
    #                 s_n = np.sqrt(np.sum((best-VAMM)**2)/(self.n_exe-1))
    #                 # ME al 95%
    #                 self.ME[t] = 2*s_n/np.sqrt(self.n_exe)
    
                   
    #                 # Muestra el progreso
    #                 print("Generación {}: \n VAMM={} ME = {} \n ".format(
    #                       it, self.VAMM[it], self.ME[t]))
                
    #                 print("Generación {}: \n VAMM={} ME = {} \n ".format(
    #                       it, self.VAMM[it], self.ME[t]),
    #                        file=log)
    #                 t+=1
    #             for i in range(self.n_exe):
    #                 #Condición de convergencia
                    
    #                 if(
    #                    not self.exito[i] # si no se ha llegado a convergencia aun gen_converg=0
    #                    and abs(self.sample[i].bestadapt-prev_best[i,it%100]) < 0.01
    #                 ):
    #                        self.exito[i]=True
    #                        t2 = time.time()
    #                        self.T_conver[i]=t2-t1
    #                        print("-------CONVERGENCIA EJECUCIÓN {} (GENERACIÓN {})------".format(
    #                                                                                        i, it))
    #                        print("-------CONVERGENCIA EJECUCIÓN {} (GENERACIÓN {})------".format(
    #                                                                                        i, it),
    #                              file=log)     
                    
    #             prev_best[:,it%100]=[self.sample[i].bestadapt for i in range(self.n_exe)]
    #         #FIN DEL BUCLE.    
    #         # Ahora, mide el tiempo, guarda la generación de
    #         # convergencia y plotea la grafica de progreso
    #         t2 = time.time()
    #         print("\nTiempo={}s".format(t2-t1))
    #         print("\nTiempo={}s".format(t2-t1), file=log)
            
                            
            
    #         print("\nMEJOR RUTA (COSTE={}):\n{}".format(
    #                                 best_pop.bestadapt,
    #                                 best_pop.pop[best_pop.best]),
    #                                 file=log)
    #         self.plot_generations()
    #         return TE, self.VAMM, self.ME, TM
     
       


        
        
        
        #%%############### FUNCIONES PARA CARGAR LOS PARÁMETROS Y CIUDADES #######################



    def load_params(self):
        """Carga los parámetros desde param_file. Si param_file=None, carga los 
        parámetros por defecto.

        param_file: String con al ubicación del archivo json de parámetros"""
        try:
            print("Cargando parametros de fichero.\n") 
            with open(self.param_file) as fp:
                params = json.load(fp)
        # Si la ubicación de param_file no es valida, carga la configuración 
        #por defecto
        except:
            print("Fichero de parametros invalido/vacio.\n",
                  "Cargando parametros por defecto.\n")  
            params = {}
            params["ngen"] = 100
            params["npop"] = 10
            params["ps"] = 1
            params["t_size"] = 2
            params["n_torneos"] = params["npop"]
            params["pc"] = 0.2
            params["pm"] = 1
            self.params=params   
            self.save_params()
              
        return params
    
    def save_params(self):
        # Codigo para guardar los parametros
        with open(self.param_file,"w") as fp:
                  json.dump(self.params, fp)
        
    def modify_parameters(key,value,self):
         self.parameters[key]=value
         self.save_params()
         
         
    def load_instance(self):
        """
        Carga las ciudades y el óptimo conocido de la instancia del problema
        dada por self.instance (ver __init__). 

        Ambas instancias vienen dadas por archivos .tsp extraídas de la página
        http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/index.html

        La instancia facil es el problema "eil101" de dicha página y la  
        la instancia compleja es "rl11849".
        """

        # Carga las ciudades de los archivos .tsp en una numpy array.

        # (Las indexaciones [6:-1] y [:,1:] son necesarias para cargar las
        # coordenadas de las ciudades corresctamente, por las descripciones, las
        # lineas en blanco y la numeración en el archivo .tsp)

        # Hemos añadido el óptimo (extraido de la página) manualmente. El  
        # optimo de la instancia compleja es solo una cota inferior del óptimo.
        # real. Dado que, en geenral, el algoritmo queda lejos del optimo,
        # no usamos este dato para la condición de terminación, es solo 
        # a nivel informativo

        if self.instance == "simple":
            with open("./instances/simple.tsp") as fp:
                f = fp.readlines()
                cities = np.loadtxt(f[6:-1], dtype=int)[:, 1:]
            optimal = 629
        elif self.instance == "complex":
            with open("./instances/complex.tsp") as fp:
                f = fp.readlines()
                cities = np.loadtxt(f[6:-1], dtype=int)[:, 1:]
            optimal = 923368
        else:
            raise ValueError("Selecciona Instance=\"simple\" ó \"complex\"")
        return cities, optimal


#%%######################################## MAIN #########################################
if __name__ == "__main__":
    n_exe=30
    T = Test(name="pc_0.2", instance="complex", n_exe=n_exe)
    #T.initial_evaluation()
    T.run()
    
    
    
    plt.figure(figsize=(8, 6),dpi=200)
    T.addplot_VAMM("pc_0.2")
    plt.grid(True)
    plt.xlim(0, T.params["ngen"])
    plt.legend()
    plt.xlabel('Generaciones')
    plt.ylabel('Valor de adaptación')
    plt.title('Evaluacion AG: instancia {},  {}. n_exe={}, T={}'.format(
                                                    T.instance, T.name, n_exe, T.T_exe))
    plt.grid(True)

    plt.show()
    # A.execute(100)