from Data import Data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


classifiers = [
    DecisionTreeClassifier(max_depth=17),
    GaussianNB(),
    AdaBoostClassifier(), # 2
    MLPClassifier(activation="logistic", solver="sgd", learning_rate="adaptive"),
    MLPClassifier(activation="tanh", solver="sgd", learning_rate="adaptive"),
    MLPClassifier(activation="logistic", learning_rate="adaptive"),
    MLPClassifier(activation="tanh", learning_rate="adaptive"), # 6
    QuadraticDiscriminantAnalysis(),

    RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    SVC(gamma=2, C=1)
]

trainData = Data()
data = trainData.readTrainData()

trainData.predict(classifiers[8], 'RandomForest', '')
