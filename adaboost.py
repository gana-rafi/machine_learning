from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import adaclass
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from adaclass import AdaBoost
mpl.use('Qt5Agg')
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_()) 


def make_toy_dataset(n: int = 100, random_seed: int = None) -> (np.ndarray, np.ndarray):
    """ Generate a toy dataset for evaluating AdaBoost classifiers

    Source: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

    """

    n_per_class = int(n/2)

    if random_seed:
        np.random.seed(random_seed)

    # the X[i] is the sample
    # the y[i] is the result of the sample
    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)

    return X, y*2-1


def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
    """ Fit the model using training data """

    X, y = self._check_X_y(X, y)
    n = X.shape[0]

    # init numpy arrays
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)

    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n

    for t in range(iters):
        # fit  weak learner
        curr_sample_weights = self.sample_weights[t]
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump = stump.fit(X, y, sample_weight=curr_sample_weights)
		
        # calculate error and stump weight from weak learner prediction
        stump_pred = stump.predict(X)
        err = curr_sample_weights[(stump_pred != y)].sum()  # / n
        stump_weight = np.log((1 - err) / err) / 2

        # update sample weights
        new_sample_weights = curr_sample_weights * \
            np.exp(-stump_weight * y * stump_pred)
        new_sample_weights /= new_sample_weights.sum()

        # If not final iteration, update sample weights for t+1
        if t+1 < iters:
            self.sample_weights[t+1] = new_sample_weights

        # save results of iteration
        self.stumps[t] = stump
        self.stump_weights[t] = stump_weight
        self.errors[t] = err

    return self


def predict(self, X):
    """ Make predictions using already fitted model """
    stump_preds = np.array([stump.predict(X) for stump in self.stumps])
    return np.sign(np.dot(self.stump_weights, stump_preds))


def truncate_adaboost(clf, t: int):
    """ Truncate a fitted AdaBoost up to (and including) a particular iteration """
    assert t > 0, 't must be a positive integer'
    from copy import deepcopy
    new_clf = deepcopy(clf)
    new_clf.stumps = clf.stumps[:t]
    new_clf.stump_weights = clf.stump_weights[:t]
    return new_clf


def plot_staged_adaboost(X, y, clf, iters=10):
    """ Plot weak learner and cumulaive strong learner at each iteration. """

    # larger grid
    fig, axes = plt.subplots(figsize=(8, iters*3),
                             nrows=iters,
                             ncols=2,
                             sharex=False,
                             dpi=100)

    fig.set_facecolor('white')

    _ = fig.suptitle('Decision boundaries by iteration')
    for i in range(iters):
        ax1, ax2 = axes[i]

        # Plot weak learner
        _ = ax1.set_title(f'Weak learner at t={i + 1}\n\u03B1={clf.stump_weights[i]}\n\u03B5={clf.errors[i]}')
        plot_adaboost(
            X, y, clf.stumps[i], sample_weights=clf.sample_weights[i], annotate=False, ax=ax1)

        # Plot strong learner
        trunc_clf = truncate_adaboost(clf, t=i + 1)
        _ = ax2.set_title(f'Strong learner at t={i + 1}\n\u03B1={clf.stump_weights[i]}\n\u03B5={clf.errors[i]}')
        plot_adaboost(
            X, y, trunc_clf, sample_weights=clf.sample_weights[i], annotate=False, ax=ax2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    a = ScrollableWindow(fig)

AdaBoost.fit = fit
AdaBoost.predict = predict
X, y = make_toy_dataset(n=10, random_seed=10)
# here we have a fake dataset
clf = AdaBoost().fit(X, y, iters=20)
plot_staged_adaboost(X, y, clf, iters=20)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.1%}')
