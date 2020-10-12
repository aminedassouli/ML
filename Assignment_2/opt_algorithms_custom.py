import numpy as np
import time 
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from sklearn.metrics import accuracy_score
from mlrose.neural import flatten_weights, unflatten_weights


def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None, curve=False, random_state=None):
    """Use randomized hill climbing to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Brownlee, J (2011). *Clever Algorithms: Nature-Inspired Programming
    Recipes*. `<http://www.cleveralgorithms.com>`_.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -1*np.inf
    best_state = None
    state = []
    times = []
    iterations = []

    if curve:
        fitness_curve = []

    for _ in range(restarts + 1):
        # Initialize optimization problem and attempts counter
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0
        start = time.time()

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)
            

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter


            if next_fitness > problem.get_fitness():
                #print(next_state)
                #print('improvement')
                attempts = 0
                problem.set_state(next_state)


            else:
                attempts += 1

            state.append(problem.get_state())
            iterations.append(iters)
            current_time = time.time()
            times.append(current_time-start)

            if curve:
                fitness_curve.append(problem.get_fitness())

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness
    #print('state')
    #print(state)
    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve), state, times, iterations

    return best_state, best_fitness, state, times, iterations








def simulated_annealing(problem, schedule=mlrose.decay.GeomDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None, curve=False,
                        random_state=None):
    """Use simulated annealing to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    if curve:
        fitness_curve = []

    attempts = 0
    iters = 0
    state = []
    times = []
    iterations = []
    start = time.time()
    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter

            if (delta_e > 0) or (np.random.uniform() < prob):
                
                attempts = 0
                problem.set_state(next_state)


            else:
                attempts += 1

            current_time = time.time()
            state.append(problem.get_state())
            iterations.append(iters)
            times.append(current_time-start)

        if curve:
            fitness_curve.append(problem.get_fitness())

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve), state, times, iterations

    return best_state, best_fitness, state, times, iterations





def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf, curve=False, random_state=None):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array of arrays containing the fitness of the entire population
        at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if curve:
        fitness_curve = []

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0
    state = []
    times = []
    iterations = []
    start = time.time()
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter

        if next_fitness > problem.get_fitness():
            attempts = 0
            problem.set_state(next_state)


        else:
            attempts += 1


        current_time = time.time()
        state.append(problem.get_state())
        iterations.append(iters)
        times.append(current_time-start)


        if curve:
            fitness_curve.append(problem.get_fitness())

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve), state, times, iterations

    return best_state, best_fitness, state, times, iterations







def gradient_descent(problem, max_attempts=10, max_iters=np.inf,
                     init_state=None, curve=False, random_state=None):
    """Use gradient_descent to find the optimal neural network weights.
    Parameters
    ----------
    problem: optimization object
        Object containing optimization problem to be solved.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        Numpy array containing starting state for algorithm.
        If None, then a random state is used.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    if curve:
        fitness_curve = []

    attempts = 0
    iters = 0
    state = []
    times = []
    iterations = []
    start = time.time()

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Update weights
        updates = flatten_weights(problem.calculate_updates())

        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        if next_fitness > problem.get_fitness():
            attempts = 0
        else:
            attempts += 1

        current_time = time.time()
        state.append(problem.get_state())
        iterations.append(iters)
        times.append(current_time-start)

        if next_fitness > problem.get_maximize()*best_fitness:
            best_fitness = problem.get_maximize()*next_fitness
            best_state = next_state

        if curve:
            fitness_curve.append(problem.get_fitness())

        problem.set_state(next_state)


    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve), state, times, iterations

    return best_state, best_fitness, state, times, iterations


