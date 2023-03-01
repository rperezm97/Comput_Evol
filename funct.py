# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:21:08 2023

@author: berti
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the abstract class "Function"
class Function:
    """
    Class representing a mathematical n-dimensional function with a name and a 
    domain.

    Attributes:
        name (str): The name of the function.
        dom (tuple): The domain of the function as a tuple. Default value is 
        None (which represents that the domain real hyperplane).

    Methods:
        function (x): Abstract method that must be overridden in child classes 
        to specify the implementation of the function.
        plot(): Plots the surface of the function for n=2 in a 3D plot.
    """
    def __init__(self, name, dom=None):
        """
        Constructor for the Function class.
        """
        self.name = name
        self.dom = dom
    
    def function(self, x):
        """
        Abstract method that must be overridden in child classes to specify the
        implementation of the function.
        """
        pass
    
    def plot(self):
        """
        Plots the surface of the function (for n=2) in a 3D plot, inside its 
        domain.
        """
        # Create a meshgrid in the domain, with a resolution of 200x200. If the
        #domain is the real hyperplane, we'll plot in between (-50,50)
        if self.dom:
            dom=self.dom
        else:
            dom=(-50,50)
        x = np.linspace(*dom, num=200)
        y = np.linspace(*dom, num=200)
        X, Y = np.meshgrid(x, y)
        #Evaluate the function on the points of the grid (for the, first we put
        #The points of the grid in a 2D vector and the we reshape the result
        #again to match the grid)
        Z = self.function(np.column_stack((X.ravel(), Y.ravel())))
        Z = Z.reshape(X.shape)
        
        # Plot the surface, with a color map to see the maxima and the minima
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap="plasma")
        ax.set_title(self.name)
        plt.show()

# Define the child class "Powell_sum"
class Powell_sum(Function):
    """
    Class representing the Powell sum function. Inherits from the Function 
    class.

    Methods:
        function (x): Specifies the implementation of the Powell sum function.
    """
    def __init__(self):
        """
        Constructor for the Powell_sum class.
        """
        # Call the constructor of the parent class 
        super().__init__("Powell Sum", dom=(-1,1))
    
    def function(self, x):
        """
        Specifies the implementation of the Powell sum function.
        
        Parameters
        ----------
        x : np.array or list with k n-dimensional points where to 
        evaluate the function. Also can be one ndarray representing one point
        
        Returns
        -------
        np.array of dimension k, or float if the input is just one point
        """
        # We need the dimension of the points /the point
        n = x.shape[1] if len(x.shape)==2 else x.shape[0]
        return np.sum(np.abs(x)**(np.arange(1, n+1) + 1), axis=1)

# Define the child class "Xin_she_yang_4"
class Xin_she_yang_4(Function):
    """
    Class to implement Xin-She Yang's Function No.04.
    This class is a child class of the class Function.
    """
    
    def __init__(self):
        """
        Constructor of the class Xin_she_yang_4.
        """
        # Call the constructor of the parent class 
        super().__init__("Xin-She Yang's Function No.04", dom=(-10,10))
    
    def function(self, x):
        """
        Implements Xin-She Yang's Function No.04.
        
        Parameters:
        x : np.array or list with k n-dimensional points where to 
        evaluate teh function
        
        Returns:
        np.array of dimension k (since its a scalar function)
        """
        # Define the dimension of the array the represents the variables
        a=len(x.shape)-1
        return np.sum(np.sin(x)**2 , axis=a)- np.exp(-np.sum(x**2, axis=a))* \
            np.exp(-np.sum(np.sin(np.sqrt(abs(x)))**2, axis=a))

    
if __name__ == "__main__":
    
    powell = Powell_sum()
    powell.plot()
    
    xin_she_yang = Xin_she_yang_4()
    xin_she_yang.plot()