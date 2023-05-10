
def load_parameters(parameters_file):
    """
    Load the parameters from the json file specified in param_file. 
    If param_file is None or the file is not found, default parameters will 
    be used.

    :return: A dictionary containing the parameters.
    """
    try:
        print("Loading parameters from file.\n")
        with open(parameters_file) as fp:
            parameters = json.load(fp)
    except:
        print("Invalid/empty parameter file.\nUsing default parameters.\n")
        parameters = {}
        parameters["n_gen"] = 1000
        parameters["n_pop"] = 1000
        parameters["ps"] = 1
        parameters["t_size"] = 3
        parameters["n_tournaments"] = parameters["n_pop"]
        parameters["pc"] = 1
        parameters["pm"] = 1
        parameters["elitism"]=1
        # save_parameters(parameters, "parmeters/default.json")
    return parameters

def save_parameters(parameters,parameters_file):
    """
    Save the parameters to the json file specified in param_file.
    """
    with open(parameters_file, "w") as fp:
        json.dump(parameters, fp)

def modify_parameters(parameters, key, value):
    """
    Modify a parameter and save it to the parameter file.

    :param key: The parameter to be modified.
    :param value: The new value for the parameter.
    """
    parameters[key] = value
    save_parameters()
