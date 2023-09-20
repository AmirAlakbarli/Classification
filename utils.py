import numpy as np
import matplotlib.pyplot as plt


def generate_univariate_x(m):
    x = 20 * np.random.rand(m) - 10
    x = np.sort(x)
    return x


def generate_multivariate_x(*shape):
    X = 10 * np.random.rand(*shape)
    # sort by Euclidean distance from origin
    X = X[np.argsort(np.sum(X**2, axis=1))]
    return X


def generate_binary_y(m):
    y_false = np.zeros(int(m * 0.45))
    y_true = np.ones(int(m * 0.45))
    y_mixed = np.random.randint(0, 2, m - len(y_false) - len(y_true))
    y = np.concatenate((y_false, y_mixed, y_true))
    return y


def generate_multiclass_y(m):
    y0 = np.zeros(int(m*0.23))
    y0_y1_mixed = np.random.randint(0, 2, int(m * 0.03))
    y1 = np.ones(int(m*0.23))
    y1_y2_mixed = np.random.randint(1, 3, int(m * 0.03))
    y2 = np.ones(int(m*0.23)) * 2
    y2_y3_mixed = np.random.randint(2, 4, int(m * 0.03))
    m = m - len(y0) - len(y1) - len(y2) - len(y0_y1_mixed) - \
        len(y1_y2_mixed) - len(y2_y3_mixed)
    y3 = np.ones(m) * 3
    y = np.concatenate((y0, y0_y1_mixed, y1, y1_y2_mixed, y2, y2_y3_mixed, y3))
    return y


def f1_score(y_train, y_predicted):
    tp = np.sum(y_train[y_predicted == 1] == 1)
    fp = np.sum(y_train[y_predicted == 1] == 0)
    fn = np.sum(y_train[y_predicted == 0] == 1)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def accuracy_score(y_train, y_predicted):
    return np.sum(y_train == y_predicted) / len(y_train)


def class_info(y):
    classes = np.unique(y)
    classes = np.sort(classes)
    class_num = len(classes)
    return classes, class_num


def get_borders(classes):
    class_num = len(classes)
    y_borders = np.array([(classes[i] + classes[i+1]) /
                         2 for i in range(class_num - 1)])
    return y_borders


def classify(y, borders, classes):
    class_num = len(classes)
    for i in range(class_num - 2):
        y[(y >= borders[i]) &
          (y < borders[i + 1])] = classes[i + 1]

    y[y < borders[0]] = classes[0]
    y[y >= borders[-1]] = classes[-1]
    return y


def plt_binary_classification(x, y):
    plt.scatter(x[y == 0], y[y == 0], marker='o', c='b')
    plt.scatter(x[y == 1], y[y == 1], marker='x', c='r')


def plt_3d():
    fig = plt.figure(figsize=(10, 5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # Plot configuration

    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(15, -120)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    return ax


def plt_3d_binary_classification(x, y):
    ax = plt_3d()

    ax.scatter(x[y == 0, 0], x[y == 0, 1], y[y == 0], marker='o', c='b')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], y[y == 1], marker='x', c='r')

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    return ax


def plt_3d_multiclass_classification(x, y):
    ax = plt_3d()

    ax.scatter(x[:, 0], x[:, 1], y, cmap='viridis', c=y)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    return ax
