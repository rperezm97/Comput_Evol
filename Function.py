import matplotlib.pyplot as plt
import numpy as np


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
    def __init__(self, name, function, dom=None):
        """
        Constructor for the Function class.
        """
        self.name = name
        self.function=function
        self.dom = dom
    
    def plot3D(self, dim=3):
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
        fig = plt.figure()

        if dim==3:
            
            y = np.linspace(*dom, num=200)
            X, Y = np.meshgrid(x, y)
            #Evaluate the function on the points of the grid (for the, first we put
            #The points of the grid in a 2D vector and the we reshape the result
            #again to match the grid)
            Z = self.function(np.column_stack((X.ravel(), Y.ravel())))
            Z = Z.reshape(X.shape)
            
            # Plot the surface, with a color map to see the maxima and the minima
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap="plasma")
        elif dim==2:
            y= self.function(x)
            ax = fig.add_subplot(111)
            ax.plot(x,y)
        
        ax.set_title(self.name)
        plt.show()
        
    def get_equidistant_sample(self.n_points):
        