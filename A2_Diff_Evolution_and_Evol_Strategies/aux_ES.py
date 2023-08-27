import numpy as np


def discrete_crossover(matched_parents):
    n_children, n_parents, n = matched_parents.shape
    # Randomly select parent indices for donating each variable,
    # using numpy methods for handling all offsprings simultaneously:
    # parent_var_selection[i,j] = parent index that will donate the j-th variable
    # in the i-th children generation
    parent_var_selection = np.random.randint(low=0, high=n_parents,
                                             size=(n_children, n))
    # Since , using advanced indexing
    # generate x_children[i, j] = parent_matches[i, parent_var_selection[i, j], j],7
    i = np.arange(n_children).reshape(-1, 1)
    j = np.arange(n)
    offsprings = matched_parents[i, parent_var_selection, j]
    return offsprings


def inter_crossover(matched_parents):
    offsprings = np.mean(matched_parents, axis=1)
    return offsprings


def one_step_mutation(children, tau, eps_0, dom):
    n_children, n_dims = children.shape
    n_dims -= 1

    mutated_children = np.empty(children.shape)

    # For the strategy parameter mutations, generate a vector of
    # perturbations from a normal N(0,1), and multiply
    # it by tau. Then exponentiate this to get the log normal.
    mutated_children[:, n_dims] *= np.exp(tau
                                          * np.random.normal(
                                              size=(n_children,)
                                          )
                                          )

    mutated_children[:, n_dims] = np.clip(mutated_children[:, n_dims],
                                          a_max=None,
                                          a_min=eps_0)

    # For the variable mutations, generate a matrix of perturbations
    # from a normal N(0,1) and multiply each row by the corresponding
    # sigma value
    mutated_children[:, :n_dims] += (mutated_children[:, n_dims:]
                                     * np.random.normal(
        size=(n_children, n_dims)
    ))

    mutated_children[:, :n_dims] = np.clip(mutated_children[:, :n_dims],
                                           a_min=dom[0],
                                           a_max=dom[1])
    return mutated_children


def n_step_mutation(children, tau, tau_prime, eps_0, dom):
    n_children, n_dims = children.shape
    # We divide by two, since the
    n_dims //= 2

    mutated_children = np.empty(children.shape)

    # For the strategy parameter mutations, generate the global and
    # individual normal perturbations with tau and tau', sum them
    # with numpy broadcast and exponentiate them to get the total
    # perturbation
    global_mut = tau_prime * np.random.normal(size=(n_children, 1))
    local_mut = tau * np.random.normal(size=(n_children, n_dims))
    mutated_children[:, n_dims:] *= np.exp(global_mut+local_mut)

    mutated_children[:, n_dims:] = np.clip(mutated_children[:, n_dims:],
                                           a_max=None,
                                           a_min=eps_0)

    # For the variable mutations, generate a list of perturbation
    # vectors, where each vector comes from a multivariate N(0,C),
    # where C is the diagonal covariant matrix defined by the strategy
    # parameters
    mutated_children[:, :n_dims] += [np.random.multivariate_normal(
        mean=np.zeros(n_dims),
        cov=np.diag(mutated_children[i, n_dims:]),
        size=1)[0]
        for i in range(n_children)]

    mutated_children[:, :n_dims] = np.clip(mutated_children[:, :n_dims],
                                           a_min=dom[0],
                                           a_max=dom[1])

    return mutated_children
