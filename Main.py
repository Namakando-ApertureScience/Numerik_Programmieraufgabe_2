from Methods_imports import *
from sklearn import (datasets as sets,
                     model_selection as ms)

input_ = input("If you want to use scikit-learn data press s, \n"
               "if you want to use MNIST data press m: ").lower()

if input_ == 's':
    ####################################################################################################################
    # Neural network for scikit-learn #
    ###################################

    # Network parameters
    layers = [64, 32, 16, 3]
    activation_functions = ["Hyperbolic_tangent", "Hyperbolic_tangent",
                            "Softplus"]  # Approximation_interval, Identity, Sigmoid, ReLU, Leaky_ReLU, Softplus, Hyperbolic_tangent, Softmax
    optimizers = "Adam"  # Backpropagation or Adam
    loss_function = "Cross_entropy"  # MSE or Cross_entropy
    learning_rate = 0.01
    weight_decay_rate = 0.0001

    network = Construct(layers,
                        activation_functions,
                        optimizers,
                        weight_decay_rate=weight_decay_rate,
                        learning_rate_adam=learning_rate,
                        Loss_function=loss_function)

    ####################################################################################################################
    # Training / Testing for scikit-learn #
    #######################################

    SEED = 42
    epochs = 200
    batch_size = 5
    test_size = 0.2

    np.random.seed(SEED)

    X_raw, y_raw = sets.load_digits().data, sets.load_digits().target
    mask = np.isin(y_raw, [1, 5, 7])

    X_clean, y_clean = ([X_raw[index] for index in range(len(X_raw)) if mask[index]],
                        [label_map[entry] for entry in y_raw if entry in [1, 5, 7]])

    X_train, X_test, y_train, y_test = ms.train_test_split(X_clean,
                                                           y_clean,
                                                           test_size=test_size,
                                                           random_state=SEED,
                                                           stratify=y_clean)

    X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

    X_train, y_train, X_test, y_test = (np.array(X_train), np.array(y_train),
                                        np.array(X_test), np.array(y_test))

    train_full_bath(X_train, y_train, X_test, y_test, network, epochs, batch_size)

elif input_ == 'm':

    ####################################################################################################################
    # Neural network for MNIST #
    ############################

    # Network parameters
    layers = [28 ** 2, 64, 32, 3]
    activation_functions = ["Hyperbolic_tangent", "Hyperbolic_tangent",
                            "Softplus"]  # Approximation_interval, Identity, Sigmoid, ReLU, Leaky_ReLU, Softplus, Hyperbolic_tangent, Softmax
    optimizers = "Adam"  # Backpropagation or Adam
    loss_function = "Cross_entropy"  # MSE or Cross_entropy
    learning_rate = 0.005
    weight_decay_rate = 0.0001

    network = Construct(layers,
                        activation_functions,
                        optimizers,
                        weight_decay_rate=weight_decay_rate,
                        learning_rate_adam=learning_rate,
                        Loss_function=loss_function)

    ####################################################################################################################
    # Training / Testing for MNIST #
    ################################

    SEED = 42
    epochs = 300
    batch_size = 5
    main_subpath = "/home/namakando/PycharmProjects/Numerik_Programmieraufgabe_2/MNIST/"

    np.random.seed(SEED)

    X_train, y_train, X_test, y_test = (
        load_mnist_images(main_subpath + "train-images-idx3-ubyte"),
        load_mnist_labels(main_subpath + "train-labels-idx1-ubyte"),
        load_mnist_images(main_subpath + "t10k-images-idx3-ubyte"),
        load_mnist_labels(main_subpath + "t10k-labels-idx1-ubyte")
    )

    X_train, X_test = (X_train.reshape([len(X_train), 28 ** 2]),
                       X_test.reshape([len(X_test), 28 ** 2]))

    mask_train, mask_test = (np.isin(y_train, [1, 5, 7]),
                             np.isin(y_test, [1, 5, 7]))

    X_train_filtered, y_train_filtered = ([X_train[index] for index in range(len(X_train)) if mask_train[index]],
                                          [label_map[entry] for entry in y_train if entry in [1, 5, 7]])

    X_test_filtered, y_test_filtered = ([X_test[index] for index in range(len(X_test)) if mask_test[index]],
                                        [label_map[entry] for entry in y_test if entry in [1, 5, 7]])

    X_train, X_test = scaler.fit_transform(X_train_filtered), scaler.fit_transform(X_test_filtered)

    X_train, y_train, X_test, y_test = (np.array(X_train), np.array(y_train_filtered),
                                        np.array(X_test), np.array(y_test_filtered))

    train_full_bath(X_train, y_train, X_test, y_test, network, epochs, batch_size)  # DATA SWITCHED !!!!!!!!!

else:
    raise Exception("Error! Invalid input!")
