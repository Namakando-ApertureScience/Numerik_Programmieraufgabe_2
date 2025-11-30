from Network_Constructor import *
from matplotlib import pyplot as plt
from sklearn import (preprocessing as pp,
                     metrics as met)


########################################################################################################################
# Utilities #
#############

# Image data loader
def load_mnist_images(path):
    with open(path, "rb") as f:
        # 4 big-endian int32: magic, num, rows, cols
        magic, num, rows, cols = np.fromfile(f, dtype=">i4", count=4)
        assert magic == 2051
        data = np.fromfile(f, dtype=np.uint8)
    return data.reshape(num, rows, cols)


# Label data loader
def load_mnist_labels(path):
    with open(path, "rb") as f:
        # 2 big-endian int32: magic, num
        magic, num = np.fromfile(f, dtype=">i4", count=2)
        assert magic == 2049
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def plot(epochs, training_acc, testing_acc, rand_acc, min_test, max_test):
    plt.figure(figsize=(10, 6), facecolor="lightblue")
    plt.title("Plotted Data")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy in %")

    plt.scatter(epochs, training_acc, s=5, c="red", label="Training accuracy")
    plt.scatter(epochs, testing_acc, s=5, c="blue", label="Testing accuracy")
    plt.scatter(epochs, rand_acc, s=5, c="green", label="Random accuracy")

    plt.scatter([], [], c="white", label=f"Min test acc: {min_test[0]:.1f}% | Iteration: {min_test[1]}")
    plt.scatter([], [], c="white", label=f"Max test acc: {max_test[0]:.1f}% | Iteration: {max_test[1]}")

    plt.legend(prop={"size": 8})


def confusion(y_true, y_pred, classes):
    cm = met.confusion_matrix(y_true, y_pred, labels=classes)
    disp = met.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=15, pad=20)
    plt.xlabel("Prediction", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    plt.gca().xaxis.set_label_position("top")
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, "Prediction", ha="center", fontsize=13)


label_map = {1: 0, 5: 1, 7: 2}
label_map_ = {0: 1, 1: 5, 2: 7}

one_hot = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)}
scaler = pp.StandardScaler()


####################################################################################################################
# Training methods #
####################

def acc(func_, X_, y_):
    return met.accuracy_score(np.array([func_(x) for x in X_]), y_)


def train_full_bath(X_train_, y_train_, X_test_, y_test_, network, iterations, batch_size):
    print(f"\nSize of training set: {len(X_train_)} \n"
          f"Size of testing set: {len(X_test_)}", end="\n\n")

    epochs, training_acc, testing_acc, rand_acc = [], [], [], []
    min_, max_ = [float("inf"), 0], [0, 0]

    for i in range(iterations):
        func = lambda x: network.forward(x).argmax()
        rd = lambda x: np.random.choice(3)

        comp_train = acc(func, X_train_, y_train_)
        comp_test = acc(func, X_test_, y_test_)
        comp_rand = acc(rd, X_test_, y_test_)

        epochs.append(i)
        training_acc.append(comp_train * 1e2)
        testing_acc.append(comp_test * 1e2)
        rand_acc.append(comp_rand * 1e2)

        if min_[0] > 1e2 * comp_test:
            min_[0], min_[1] = comp_test * 1e2, i

        if max_[0] < 1e2 * comp_test:
            max_[0], max_[1] = comp_test * 1e2, i

        print(f"Iterations: {i + 1}/{iterations} |"
              f"Training accuracy: {comp_train * 1e2:.1f}% |"
              f"Testing accuracy: {comp_test * 1e2:.1f}% |"
              f"Random of test accuracy: {comp_rand * 1e2:.1f}%", end="\r")

        indices = np.random.choice(len(X_train_), size=batch_size, replace=False)
        Data0, Data1 = X_train_[indices], np.array([one_hot[y] for y in y_train_[indices]])
        network.Optimize(Data0, Data1)

    plot(epochs, training_acc, testing_acc, rand_acc, min_, max_)
    confusion(y_true=[label_map_[y] for y in y_test_],
              y_pred=[label_map_[network.forward(x).argmax()] for x in X_test_],
              classes=[1, 5, 7])
    plt.show()
