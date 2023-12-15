
"""
Created on Wed Feb  8 12:21:08 2023

@author: berti

"""
#%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC

class Function(ABC):
    def __init__(self, domain=None):
        """
        Constructor for the abstract Function class, representing a multivarible
        real valued function to be optimized in the Genetic_algoritm class
        in GA.py
        """
        self.domain = domain
        self.optimal=None
        
    def evaluate(x):
        pass
    
    def plot_function(self):
        """
        Plots the surface of the function (for n=2) in a 3D plot, inside its 
        domain
        """
        # Create a meshgrid in the domain, with a resolution of 200x200. If the
        #domain is the real hyperplane, we'll plot in between (-50,50)
        if self.domain:
            dom=self.domain
        else:
            dom=(-50,50)    
        
        fig = plt.figure(figsize=(6,6))
        x = np.linspace(*dom, num=100)
        y = np.linspace(*dom, num=100)
        X, Y = np.meshgrid(x, y)
        
     
        #Evaluate the function on the points of the grid (for the, first we put
        #The points of the grid in a 2D vector and the we reshape the result
        #again to match the grid)
        points = np.stack([X, Y] , 
                          axis=-1
                          ).reshape(-1,2)
        Z = self.evaluate(points)
        Z = Z.reshape(X.shape)
        
        # Plot the surface, with a color map to see the maxima and the minima
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap="plasma", zorder=0,alpha=1)
       
        # Set the viewing angle to view the axis from above (azim=90, elev=90)
        ax.view_init(azim=90, elev=90)
        #ax.set_zlim([-1, 3])
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('f(x)')
        
        ax.set_title(f"{type(self).__name__} function")
        plt.show()
    
        



# Define the child class "Powell_sum"
class Sphere(Function):
    """
    Class to implement Schwefel's Function.
    """
    def __init__(self):
        # Call the constructor of the parent class 
        super().__init__(domain=(-10,10))
        self.optimal=0
        self.optimal_xi=0
    def evaluate(self, x):
       
        return np.sum( x**2, axis=1)
    
# Define the child class "Xin_she_yang_4"
class Schwefel(Function):
    """
    Class to implement Schwefel's Function.
    """
    
    def __init__(self):
        
        # Call the constructor of the parent class         
        super().__init__( domain=(-500,500))
        self.optimal=0
        self.optimal_xi=420.9687
        
    def evaluate(self, x):
        """
        Implements Xin-She Yang's Function No.04.
        
        Parameters:
        x : np.array or list with k n-dimensional points where to 
        evaluate teh function
        
        Returns:
        np.array of dimension k (since its a scalar function)
        """
        
        
        
        # Define the dimension of the array the represents the variables
        _,n_dimensions=x.shape
        
        f_x=(418.9829 *n_dimensions 
               -np.sum( x *np.sin(  np.sqrt( abs(x) )), 
                       axis=1))
        return f_x

if __name__ == "__main__":
    
    powell = Sphere()
    powell.plot_function()
    