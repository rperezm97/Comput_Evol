# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:47:57 2022

@author: berti
"""
import time, os
import numpy as np
import matplotlib.pyplot as plt
from tests import Test




def main():
    print("Iniciando aplicación")
    time.sleep(.5)
    instance= input("Selecciona la instancia del problema (simple o complex): ")
    time.sleep(.5)
    print("Si fuera necesario, modifica los 3 archivos de parametros param_{}.".format(
                                                                            instance))
    time.sleep(.5)
    print("Iniciando ejecución inicial")
    time.sleep(.5)
    experiment={}
    for pc in [0.2,0.5,1]:
        param_file=".\params\params_{}_pc{}.json".format(instance, pc)
        print("Ejecución inicial caso:{} pc={}".format(instance, pc))
        experiment[pc]=Test(instance, n_exe=12, param_file=param_file)
        gen_converg=experiment[pc].estimate_Tconverg()
        print("Converge en la generación {}".format(gen_converg))
        time.sleep(.5)
    print("Analiza las gráficas de convergencia en la carpeta plots y decide"+ 
          "si el numero de generaciones es adecuado, demasiado grande o demasiado pequeño")
    try:
    ngen=16000#int(input("Introduce el numero de generaciones adecuado para la evaluación:"))
   

    results=[]
    for pc in [0.2,0.5,1]:
        param_file=".\params\params_{}_pc{}.json".format(instance, pc)
        print("Experimento de evaluación:{} pc={}".format(instance, pc))
        results.append(experiment[pc].execute(ngen))
        
        
        time.sleep(.5)


    plot_comparison(instance, experiment, ngen)


def plot_comparison(instance, experiment,ngen):
    
    
    ax=plt.figure(figsize=(8, 6),dpi=200)
    for pc in experiment.keys():
        experiment[pc].addplot_VAMM(ax, "pc={}".format(pc))
    plt.grid(True)
    plt.xlim(0, ngen)
    plt.legend()
    plt.xlabel('Generaciones')
    plt.ylabel('Valor de adaptación')
    plt.title('Comparación de evaluaciones AG (instancia {})'.format(
                                            instance))
    plt.grid(True)

    plt.savefig('./plots/Comparison_{}.png'.format(
                                            instance))
    plt.show()



  
if __name__ == "__main__":

    main()
    
