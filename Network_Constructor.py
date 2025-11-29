import numpy as np


def derivative(func, x, quant=1e-13):
    x = np.array(x, dtype=np.float128)
    return (func(x + quant) - func(x - quant)) / (2 * quant)


class Activation_functions:

    def __init__(self, slope, leaky_slope=0.05, approximation_interval=None):
        if approximation_interval is None:
            self.approximation_interval = [0, 1]
        else:
            self.approximation_interval = approximation_interval

        self.slope = slope
        self.leaky_slope = leaky_slope

    # Input real data gets normalised and then, put within the approximation interval
    def Approximation_interval(self, x):
        return ((self.approximation_interval[1] - self.approximation_interval[0]) *
                self.Sigmoid(x) + self.approximation_interval[0])

    # Identity (does nothing. xD)
    @staticmethod
    def Identity(x):
        return x

    # Input real data and output normalized data
    def Sigmoid(self, x):
        x = np.array(x, dtype=np.float128)
        return 1 / (1 + np.e ** (-self.slope * x))

    # Input real data output non-linear data
    def Leaky_ReLU(self, x):
        x = np.array(x, dtype=np.float128)
        return ((self.slope - self.leaky_slope) / 2) * (np.abs(x) + x) + self.leaky_slope * x

    # Input real data output non-linear data
    def ReLU(self, x):
        self.leaky_slope = 0
        return self.Leaky_ReLU(x)

    # Input real data output non-linear data
    def Softplus(self, x):
        x = np.array(x, dtype=np.float128)
        return self.slope * np.log(1 + np.exp(x))

    # Input real data and output standardised data
    def Hyperbolic_tangent(self, x):
        return (np.exp(2 * self.slope * x) - 1) / (np.exp(2 * self.slope * x) + 1)

    # Input real data and output normalised exponential data
    @staticmethod
    def Softmax(x):
        return np.exp(x) / (np.exp(x).sum())


class Construct:

    def __init__(self, layers=None, str_activation_functions=None, optimizers=None, shape=None,
                 weight_interval=None, input_output_steepness=None, weight_decay_rate=0.0001,
                 learning_rate_BP=0.1, learning_rate_adam=0.001, rho1=0.9, rho2=0.999, Loss_function="MSE"):

        ################################################################################################################
        # Neural network constructor #
        ##############################
        # Hyper parameter setting

        if layers is None and shape is None:
            raise Exception("Necessary parameters missing.")

        if layers is None:
            layers = [shape[1] for _ in range(shape[0])]

        self.layers = layers

        if input_output_steepness is None:
            self.slopes = np.full(len(layers) - 1, 1)
        else:
            self.slopes = np.linspace(input_output_steepness[0], input_output_steepness[1], len(layers) - 1)

        if str_activation_functions is None:
            str_activation_functions = ["Sigmoid" for _ in layers][:-1]
        elif type(str_activation_functions) is not list:
            str_activation_functions = [str_activation_functions for _ in layers][:-1]

        if weight_interval is None:
            weight_interval = [-0.5, 0.5]

        if optimizers is None:
            optimizers = ["Backpropagation" for _ in layers][:-1]
        elif type(optimizers) is not list:
            optimizers = [optimizers for _ in layers][:-1]

        self.optimizers = optimizers

        # Activation functions
        self.activation_functions = []

        for _, act_spec, slope in zip(layers[:-1], str_activation_functions, self.slopes):
            if isinstance(act_spec, tuple) and act_spec[0] == "Approximation_interval":
                activation = Activation_functions(
                    approximation_interval=act_spec[1],
                    slope=slope,
                ).Approximation_interval
            else:
                activation = getattr(Activation_functions(slope=slope), act_spec)

            self.activation_functions.append(activation)

        # Important matrices
        self.weighted_sums_matrix = []

        # Layer construction
        self.neural_network = []
        for index in range(1, len(layers)):
            self.neural_network.append((weight_interval[1] - weight_interval[0]) *
                                       np.random.rand(layers[index], layers[index - 1] + 1) + weight_interval[0])

        ################################################################################################################
        # Losses constructor #
        ######################

        self.Loss_function = self.MSE if Loss_function == "MSE" else (
            self.Cross_entropy if Loss_function == "Cross_entropy" else (
                None))

        ################################################################################################################
        # optimizers constructor #
        ##########################

        # Weight decay rate
        self.weight_decay_rate = weight_decay_rate

        ################################################################################################################
        # For backpropagation

        # Learning rate
        self.learning_rate_BP = learning_rate_BP

        ################################################################################################################
        # For Adam

        # Bias correction
        self.epsilon = 1e-10
        self.bias_correction = {}

        # Learning rate
        self.learning_rate_adam = learning_rate_adam

        # Time step
        self.time_step = {}
        self.time_step_bool = {}

        # Momenta
        self.rho1 = rho1
        self.rho2 = rho2
        self.AF = {}

        if "Adam" in self.optimizers:
            self.Adam_Init()

    ####################################################################################################################
    # Loss functions #
    ##################

    # Derivative
    def del_(self):
        return derivative(self.activation_functions[len(self.layers) - 2],
                          self.weighted_sums_matrix[len(self.layers) - 2])

    # MSE
    def MSE(self, X_point, y_point):
        dif = y_point - self.forward(X_point)  # First
        del_ = self.del_()  # Second (Order is important !!!)
        return dif * del_

    # Cross-entropy
    def Cross_entropy(self, X_point, y_point):
        logits = self.forward(X_point)  # First
        dif = y_point - (np.exp(logits) / (np.exp(logits).sum()))
        del_ = self.del_()  # Second (Order is important !!!)
        return dif * del_

    ####################################################################################################################
    # Optimizer Initialization #
    ############################

    # Adam
    def Adam_Init(self):
        # Momenta

        Indices = [i for i in range(len(self.optimizers)) if self.optimizers[i] == "Adam"]
        self.AF = {}

        for index in Indices:
            number_of_neurons, number_of_weights = self.layers[index + 1], self.layers[index]

            A = np.zeros((number_of_neurons, number_of_weights + 1))
            F = np.zeros((number_of_neurons, number_of_weights + 1))

            self.AF[index] = [A, F]

            self.time_step[index] = 0
            self.time_step_bool[index] = True

    ####################################################################################################################
    # optimizer #
    #############

    # Get/Set optimizers

    def Get_optimizers(self):
        return self.optimizers

    def Set_optimizers(self, optimizers):
        self.optimizers = optimizers
        self.time_step = {}
        self.time_step_bool = {}

        if "Adam" in self.optimizers:
            self.Adam_Init()

    ####################################################################################################################
    # Backpropagation

    def Backpropagation_prop_update(self, gradient, index):
        self.neural_network[index] = (self.neural_network[index] + self.learning_rate_BP * np.matmul(
            np.array([gradient]).transpose(), np.array([np.append(self.activation_functions[index](
                self.weighted_sums_matrix[index - 1]), 1)])) - self.weight_decay_rate * self.neural_network[index])

    ####################################################################################################################
    # Adam

    def Adam_update(self, gradient, index):
        grad = (np.matmul(np.array([gradient]).transpose(),
                          np.array([np.append(self.activation_functions[index]
                                              (self.weighted_sums_matrix[index - 1]), 1)])))

        self.AF[index][0] = self.rho2 * self.AF[index][0] + (1 - self.rho2) * grad ** 2
        self.AF[index][1] = self.rho1 * self.AF[index][1] + (1 - self.rho1) * grad

        self.neural_network[index] = ((self.neural_network[index] +
                                       (self.learning_rate_adam * self.bias_correction[index]) *
                                       (self.AF[index][1] / (self.AF[index][0] + self.epsilon) ** 0.5)) -
                                      self.weight_decay_rate * self.neural_network[index])

    ####################################################################################################################
    # Distributer

    def distributer(self, gradient, index=None):
        if index is None:
            index = len(self.layers) - 2

        if self.optimizers[index] == "Adam" and self.time_step_bool[index]:
            self.time_step[index] += 1
            self.bias_correction[index] = (np.sqrt(1 - self.rho2 ** self.time_step[index]) /
                                           (1 - self.rho1 ** self.time_step[index]))

            if np.abs(self.bias_correction[index] - 1) < self.epsilon:
                self.time_step_bool[index] = False
                self.bias_correction[index] = 1

        (self.Backpropagation_prop_update(gradient, index) if self.optimizers[index] == "Backpropagation"
         else (self.Adam_update(gradient, index) if self.optimizers[index] == "Adam"
               else print("Error no optimizers specified!!!")))

    ####################################################################################################################
    # Optimizer

    def Optimize(self, X, y):

        gradient = (sum(
            [self.Loss_function(X_point, y_point) for X_point, y_point in zip(X, y)]
        )) / len(X)

        self.distributer(gradient)

        for index in range(len(self.layers) - 3, 0, -1):
            gradient = (derivative(self.activation_functions[index], self.weighted_sums_matrix[index]) *
                        np.matmul(np.delete(self.neural_network[index + 1].transpose(),
                                            self.layers[index + 1], axis=0), gradient))
            self.distributer(gradient, index)

    ####################################################################################################################
    # Neural network #
    ##################

    def forward(self, input_vector):
        # Bias neuron
        input_vector = list(input_vector).copy()
        input_vector.append(1)

        self.weighted_sums_matrix = []
        self.weighted_sums_matrix.append(np.matmul(self.neural_network[0], input_vector))

        for index in range(1, len(self.neural_network)):
            self.weighted_sums_matrix.append(np.matmul(self.neural_network[index],
                                                       np.append(self.activation_functions[index - 1](
                                                           self.weighted_sums_matrix[-1]), 1)))
        return self.activation_functions[-1](self.weighted_sums_matrix[-1])
