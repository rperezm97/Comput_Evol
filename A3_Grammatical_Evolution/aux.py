
import numpy as np
import os 
import json
import random

### FUNCTIONS TO LOAD DATA

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
        parameters["n_samples"]=41
        # save_parameters(parameters, "parmeters/default.json")
    return parameters


def load_BNF_rules(rules_file):
    """
    Load the gramatical rules in the Backus-Naur notation from the json file 
    specified in rules_file. 
    If rules_file is None or the file is not found, default rules will 
    be used.

    :return: A dictionary containing the parameters.
    """
    try:
        print("Loading BNF rules from file.\n")
        with open(rules_file) as fp:
            BNF_rules = json.load(fp)
    except:
        print("Invalid/empty rules file.\nUsing default parameters.\n")
        BNF_rules = {
            "N" : {"<expr>", "<signo>", "<real>", "<K_G>","<K_P>","<K_S>", 
                   "<nulo>", "<grado>", "<uno-nueve>","<cero-nueve>"},
            "T" : {"KG", "KP", "KS", "(", ")", "+", "-", "*","NULL1", "NULL2", ",", ".",
            "E", "0","1","2","3","4","5","6", "7", "8", "9"},
            "S": "<expr>",
            
            "<expr>": [
                ["<signo>", "<real>", "*","<K_G>"],
                ["<signo>", "<real>", "*","<K_G>", "<expr>"],
                ["<signo>", "<real>", "*","<K_P>"],
                ["<signo>", "<real>", "*","<K_P>", "<expr>"],
                ["<signo>", "<real>", "*","<K_S>"],
                ["<signo>", "<real>", "*","<K_S>", "<expr>"]
            ],
            "<K_G>": [["KG", "(", "<real>", ",", "<real>",",", "<nulo>", ")"]],
            "<K_P>": [["KP", "(", "<real>",",", "<real>", ",","<grado>", ")"]],
            "<K_S>": [["KS", "(", "<real>", ",","<real>", ",","<nulo>", ")"]],
            "<signo>": [["+"] , ["-"]],
            "<real>": [["<uno-nueve>", ".", "<cero-nueve>", "E", "<signo>", "<cero-nueve>"]],
            "<nulo>": [["NULL1"], ["NULL2"]],
            "<grado>": [["0"],["1"],["2"],["3"],["4"]],
            "<uno-nueve>": [["1"],["2"],["3"],["4"],["5"],["6"],["7"],["8"],["9"]],
            "<cero-nueve>":[["0"],["1"],["2"],["3"],["4"],["5"],["6"],["7"],["8"],["9"]]
            }
    return BNF_rules

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

### FUNCTION TO DECODE CHROMOSOME

def decode(self, chromosome, BNF_rules, expression=None, decoding_idx=0):
        """
        Decodes a solution (chromosome) into a list of kernels and their weights.
        """
        if not expression:
            expression=BNF_rules["S"]
        non_terminal=False
        #Find the first non terminal element
        for i, S in enumerate(expression):
            if S in BNF_rules["N"]: 
                 non_terminal=True
                 break
        if non_terminal: 
            codon=chromosome[decoding_idx]
            S_options=BNF_rules[S]
            # If theres only one option for the symbol (ie, S is <real>, 
            # <K_G>, <K_P> or <K_S>), don't consume a codon, just update the  
            # expression expanding the symbol 
            if len(S_options)==1:
                new_expression=(expression[:i]+
                                S_options[0]+
                                expression[i+1:])
            # In the case that there are 6 options, i.e., in the expansion of 
            # an <"expr"> symbol, since we decided not to use envelopement (the 
            # chromosome is not cyclical), we have to take care that we get a
            # non finishing expression (of the form <Kernel><expr>) if there's 
            # more than 1 kernel set (15 codons) left, and a finishing 
            # expression (of the form <Kernel>) if the decoding_idx starts the 
            # last set.
            elif len(S_options)==6:
                # Since tehre are 3 kernels, and 2 options for each kernel
                # (finishing and not finishing), we use the codon to select the 
                # type of kernel, and the decoding_idx to see if it is the last 
                # set 
                option=codon%3
                is_last_set= int((len(chromosome)-decoding_idx)==15)
                
                new_expression=(expression[:i]+
                                S_options[2*option+is_last_set]+
                                expression[i+1:])
                decoding_idx=decoding_idx+1
            # Else, use the codon to chose an option, and update the decoding 
            # index to the next one (module the lenght of the chromosome, to 
            # apply envelopement)
            else:
                new_expression=(expression[:i]+
                                S_options[codon%len(S_options)]+
                                expression[i+1:])
                decoding_idx=decoding_idx+1
            print(new_expression, decoding_idx)
            # Do a recursive call to decode the new expression
            return decode(chromosome, 
                              new_expression, 
                              decoding_idx)
            
        else: 
            print("".join(expression))
            kernel=eval("".join(expression))
            return 


