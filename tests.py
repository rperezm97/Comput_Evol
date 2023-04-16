#%%#################### IMPORTS Y VARIABLES GLOBALES ##########################
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

# Usamos este string para tener un contador de generaciones.
# (Con print(end=LINE_CLEAR), se borra la ultima línea impresa en la consola).
LINE_CLEAR = '\x1b[2K'

###############################################################################
# %%           CLASE DE TESTS PARA EL ALGORITMO GENÉTICO                      #
###############################################################################

class EA_run:
    def __init__(self, EA, name, n_exe=30, output=None):
        """ 
        instance:string. Toma el valor "simple" para la instancia simple (101  
        ciudades o "complex" para la instancia compleja (100010 ciudades).

        n_exe: número de ejecuciones independientes del algoritmo para el test,
        esto es, tamaño de la muestra.

        param_file:string o None. Es la ubicación del archivo .txt con los
        valores de los parámetros del algoritmo, o None para cargar los valores
        por defecto. Ver función load_params abajo.
        
        output:  Es la ubicación del archivo donde guardar 
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

     

#################FUNCIONES PARA EJECUTAR EL TEST ################ #######
   
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