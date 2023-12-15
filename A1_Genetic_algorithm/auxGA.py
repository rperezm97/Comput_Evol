
import numpy as np
import os 
import json
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC

# ==== LOAD THE INSTANCES OF MULTIVARIABLE FUNCTIONS TO OPTIMIZE AS CLASSES ====

def load_instance(GA_instance):
    
    # Get the function name from the instance_id and initialize the corresponding
    # Function subclass
    function_name = GA_instance.instance_id.split("_")[0]
    try:
        target_f = globals().get(function_name)()
    except:
        raise ValueError(f"Multivariable Function {function_name} not defined in aux.py")
    GA_instance.target_f=target_f
    GA_instance.optimal=target_f.optimal

class Function(ABC):
    def __init__(self, domain=None):
        """
        Constructor for the abstract Function class, representing a multivarible
        real valued function to be optimized in the Genetic_algoritm class
        in GA.py
        """
        # The domain shoudl be a numpy array (n,2), where every row is 
        # [x_i_min, x_i_max]
        self.domain = domain
        self.optimal=None
        
    def evaluate(x):
        pass
    
    def plot_function(self):
        """
        Plots the surface of the function (for n=2) in a 3D plot, inside its 
        domain
        """
        # Create a meshgrid in the domain, with a resolution of 200x200. The domain
        I_x, I_y =self.domain[:2]
        
        
        fig = plt.figure(figsize=(6,6))
        x = np.linspace(*I_x, num=100)
        y = np.linspace(*I_y, num=100)
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
    
class Sphere(Function):
    """
    Class to implement Schwefel's Function.
    """
    def __init__(self, dimension):
        # Call the constructor of the parent class 
        super().__init__(domain=(-10,10))
        self.optimal=0
        self.optimal_xi=0
    def evaluate(self, x):
       
        return np.sum( x**2, axis=1)
    
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
        # Define the dimension of the array the represents the variables
        _,n_dimensions=x.shape
        
        f_x=(418.9829 *n_dimensions 
               -np.sum( x *np.sin(  np.sqrt( abs(x) )), 
                       axis=1))
        return f_x

# ================== INDIVIDUAL DECODING AND FITNESS FUNCTION ==================

def decode_genome(x, n_bits_per_dim, domain, encoding_type):

     # Create an array for the fenotypic real-valued individuals
    n_dimensions=x.shape[1]//n_bits_per_dim
    
    phenotype_x=np.zeros((x.shape[0],n_dimensions), dtype=np.float64)
    
    # Iterating through the dimensions, decode one variable at a time, 
    # simultaneously for all the population
    for j in range(n_dimensions):
        
        # Get the genes corresponding to the econding of the j-th variable
        encoded_x_j=x[:,j*n_bits_per_dim:(j+1)* n_bits_per_dim]
        
        # If the current instance uses gray encoding (separetely for each 
        # variable, see section 2.1 of the report), tranlate it into binary code
        if  encoding_type=="gray":
            encoded_x_j=gray_to_binary(encoded_x_j)
           
        # Decode the bits from x_j into the corresponding integer value 
        # from 0 to 2**n_bits_per_dim-1
        integer_x_j=np.sum(encoded_x_j * 2**np.arange(0, n_bits_per_dim), 
                                axis=1)
        
        # Translate and scale the integer values into the corresponding real-valued 
        # domain. Note that the transformation is the same for all variables
        # since the domain is an hypercube [x_o,x_f]^n_dim 
        x_o,x_f=domain
        scale=(x_f-x_o)/(2** n_bits_per_dim-1)
        phenotype_x[:,j]=x_o+ integer_x_j*scale
    return phenotype_x

def gray_to_binary(gray_array):
    """
    Convert a 2D numpy array of Gray codes to binary.

    Parameters:
    gray_array (np.array): A 2D numpy array where each row is a genotype in Gray code.

    Returns:
    np.array: A 2D numpy array where each row is the genotype in binary.
    """
    # Initialize the binary array with the same shape as the gray array
    binary_array = np.empty_like(gray_array)

    # The MSB of binary code is the same as Gray code
    binary_array[:, 0] = gray_array[:, 0]

    # Compute remaining bits
    for i in range(1, gray_array.shape[1]):
        # XOR current bit with the previous bit in binary
        binary_array[:, i] = gray_array[:, i] ^ binary_array[:, i - 1]

    return binary_array

def f_fitness(x, GA_instance):
        """
        Get the fitness of an individual (or of an array of individuals),
        by evaluating the target function in its fenotype.  
        The fenotype of an individual is a n-dimensional real valued point, 
        which is decoded applying equation 2.2 of the base book to the genotypic
        popualtion.
        """        
        
        # Since this function uses numpy vectorized methods to calculate the 
        # fitness of all teh population efficiently, if x is just one individual,
        # reshape it to form  a popualtion of 1 individual  
        if len(x.shape)==1:
            x=x.reshape((1,-1))
        phenotype_x= decode_genome(x, 
                                    GA_instance.n_bits_per_dim, 
                                    GA_instance.target_f.domain,
                                    GA_instance.encoding_type)
       
        #2. Evaluate the target function in the fenotypic population
        fitness_x= GA_instance.target_f.evaluate(phenotype_x)
        
        # If the input was 1 individual, just return its fitness value as a 
        # float. Else return a vector. 
        
        if x.shape[0]==1:
           return fitness_x[0]
        else:
           return fitness_x

# ==================== FUNCTION TO MANUALLY SAVE PARAMETERS ====================
def save_parameters(instanceName="Schwefel_G3"):
        
        parameters = {}
        parameters["n_gen"] = int(4*10e3)
        parameters["n_pop"] = 100
        parameters["n_bits_per_dim"] = 16
        parameters["n_dimensions"] = 100
        parameters["pc"] = 0.2
        parameters["pm"] =1/( parameters["n_bits_per_dim"] *
                                parameters["n_dimensions"] 
                               ) 
        parameters["sel_pressure"] = 1.7
        parameters["encoding_type"] = "gray"
        
        folder_A1=os.path.dirname(__file__)
        with open(folder_A1+"/parameters/"+instanceName+".json", 
                  "w") as fp:
            json.dump(parameters, fp)
        print(" Parameters saved: "+instanceName+".json")

if __name__=="__main__":
   
    # sphere = Sphere()
    # sphere.plot_function()
   
    save_parameters()
