import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose.neural as mn
import numpy as np
from opt_algorithms_custom import *

class NeuralNetwork(mlrose.NeuralNetwork):
    
    def _validate(self):
        if (not isinstance(self.max_iters, int) and self.max_iters != np.inf
                and not self.max_iters.is_integer()) or (self.max_iters < 0):
            raise Exception("""max_iters must be a positive integer.""")

        if not isinstance(self.bias, bool):
            raise Exception("""bias must be True or False.""")

        if not isinstance(self.is_classifier, bool):
            raise Exception("""is_classifier must be True or False.""")

        if self.learning_rate <= 0:
            raise Exception("""learning_rate must be greater than 0.""")

        if not isinstance(self.early_stopping, bool):
            raise Exception("""early_stopping must be True or False.""")

        if self.clip_max <= 0:
            raise Exception("""clip_max must be greater than 0.""")

        if (not isinstance(self.max_attempts, int) and not
                self.max_attempts.is_integer()) or (self.max_attempts < 0):
            raise Exception("""max_attempts must be a positive integer.""")

        if self.pop_size < 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(self.pop_size, int):
            if self.pop_size.is_integer():
                self.pop_size = int(self.pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        if (self.mutation_prob < 0) or (self.mutation_prob > 1):
            raise Exception("""mutation_prob must be between 0 and 1.""")

        if self.activation is None or \
           self.activation not in self.activation_dict.keys():
            raise Exception("""Activation function must be one of: 'identity',
                    'relu', 'sigmoid' or 'tanh'.""")
            
    def fit(self, X, y=None, x_test = None, y_test = None, init_weights=None):
            """Fit neural network to data.
            Parameters
            ----------
            X: array
                Numpy array containing feature dataset with each row
                representing a single observation.
            y: array
                Numpy array containing data labels. Length must be same as
                length of X.
            init_state: array, default: None
                Numpy array containing starting weights for algorithm.
                If :code:`None`, then a random state is used.
            """
            self._validate()

            # Make sure y is an array and not a list
            y = np.array(y)

            # Convert y to 2D if necessary
            if len(np.shape(y)) == 1:
                y = np.reshape(y, [len(y), 1])

            # Verify X and y are the same length
            if not np.shape(X)[0] == np.shape(y)[0]:
                raise Exception('The length of X and y must be equal.')

            # Determine number of nodes in each layer
            input_nodes = np.shape(X)[1] + self.bias
            output_nodes = np.shape(y)[1]
            node_list = [input_nodes] + self.hidden_nodes + [output_nodes]

            num_nodes = 0

            for i in range(len(node_list) - 1):
                num_nodes += node_list[i]*node_list[i+1]

            if init_weights is not None and len(init_weights) != num_nodes:
                raise Exception("""init_weights must be None or have length %d"""
                                % (num_nodes,))

            # Set random seed
            if isinstance(self.random_state, int) and self.random_state > 0:
                np.random.seed(self.random_state)

            # Initialize optimization problem
            fitness = mn.NetworkWeights(X, y, node_list,
                                     self.activation_dict[self.activation],
                                     self.bias, self.is_classifier,
                                     learning_rate=self.learning_rate)

            problem = mlrose.opt_probs.ContinuousOpt(num_nodes, fitness, maximize=False,
                                    min_val=-1*self.clip_max,
                                    max_val=self.clip_max, step=self.learning_rate)

            if self.algorithm == 'random_hill_climb':
                fitted_weights = None
                loss = np.inf
                # Can't use restart feature of random_hill_climb function, since
                # want to keep initial weights in the range -1 to 1.
                for _ in range(self.restarts + 1):
                    if init_weights is None:
                        init_weights = np.random.uniform(-1, 1, num_nodes)

                    if self.curve:
                        current_weights, current_loss, fitness_curve, state, times, iterations = random_hill_climb(problem,
                                              max_attempts=self.max_attempts if
                                              self.early_stopping else
                                              self.max_iters,
                                              max_iters=self.max_iters,
                                              restarts=0, init_state=init_weights,
                                              curve=self.curve,
                                              random_state=self.random_state)
                        #print('state')
                    else:
                        current_weights, current_loss, state, times, iterations = random_hill_climb(
                            problem,
                            max_attempts=self.max_attempts if self.early_stopping
                            else self.max_iters,
                            max_iters=self.max_iters,
                            restarts=0, init_state=init_weights, curve=self.curve,
                            random_state=self.random_state)
                    
                    if current_loss < loss:
                        fitted_weights = current_weights
                        loss = current_loss

            elif self.algorithm == 'simulated_annealing':
                if init_weights is None:
                    init_weights = np.random.uniform(-1, 1, num_nodes)

                if self.curve:
                    fitted_weights, loss, fitness_curve, state, times, iterations = simulated_annealing(
                        problem,
                        schedule=self.schedule,
                        max_attempts=self.max_attempts if self.early_stopping else
                        self.max_iters,
                        max_iters=self.max_iters,
                        init_state=init_weights,
                        curve=self.curve,
                        random_state=self.random_state)
                else:
                    fitted_weights, loss, state, times, iterations = simulated_annealing(
                        problem,
                        schedule=self.schedule,
                        max_attempts=self.max_attempts if self.early_stopping else
                        self.max_iters,
                        max_iters=self.max_iters,
                        init_state=init_weights,
                        curve=self.curve,
                        random_state=self.random_state)

            elif self.algorithm == 'genetic_alg':
                if self.curve:
                    fitted_weights, loss, fitness_curve, state, times, iterations = genetic_alg(
                        problem,
                        pop_size=self.pop_size,
                        mutation_prob=self.mutation_prob,
                        max_attempts=self.max_attempts if self.early_stopping else
                        self.max_iters,
                        max_iters=self.max_iters,
                        curve=self.curve,
                        random_state=self.random_state)
                else:
                    fitted_weights, loss, state, times, iterations = genetic_alg(
                        problem,
                        pop_size=self.pop_size, mutation_prob=self.mutation_prob,
                        max_attempts=self.max_attempts if self.early_stopping else
                        self.max_iters,
                        max_iters=self.max_iters,
                        curve=self.curve,
                        random_state=self.random_state)

            else:  # Gradient descent case
                if init_weights is None:
                    init_weights = np.random.uniform(-1, 1, num_nodes)

                if self.curve:
                    fitted_weights, loss, fitness_curve, state, times, iterations = gradient_descent(
                        problem,
                        max_attempts=self.max_attempts if self.early_stopping else
                        self.max_iters,
                        max_iters=self.max_iters,
                        curve=self.curve,
                        init_state=init_weights,
                        random_state=self.random_state)

                else:
                    fitted_weights, loss, state, times, iterations = gradient_descent(
                        problem,
                        max_attempts=self.max_attempts if self.early_stopping else
                        self.max_iters,
                        max_iters=self.max_iters,
                        curve=self.curve,
                        init_state=init_weights,
                        random_state=self.random_state)
            
            training_accuracy = []
            testing_accuracy = []
            for s in state : 
                problem.set_state(s)
                # Save fitted weights and node list
                self.node_list = node_list
                self.fitted_weights = s
                self.loss = loss
                self.output_activation = fitness.get_output_activation()
                y_train_pred = self.predict(X)
                y_train_accuracy = accuracy_score(y, y_train_pred)
                training_accuracy.append(y_train_accuracy) 
                #print('train')
                #print( y_train_accuracy)
                y_test_pred = self.predict(x_test)
                y_test_accuracy = accuracy_score(y_test, y_test_pred)
                testing_accuracy.append(y_test_accuracy)
                #print('test')
                #print(y_test_accuracy)
                #accuracy.append(problem.get_accuracy())

            if self.curve:
                self.fitness_curve = fitness_curve

            
            return self , state, training_accuracy, testing_accuracy, times, iterations
