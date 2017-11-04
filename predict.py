from Data import Data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

classifiers = [
    DecisionTreeClassifier(max_depth=10),
    GaussianNB(),
    AdaBoostClassifier(), # 2
    MLPClassifier(activation="logistic", solver="sgd", learning_rate="adaptive"),
    MLPClassifier(activation="tanh", solver="sgd", learning_rate="adaptive"),
    MLPClassifier(activation="logistic", learning_rate="adaptive"),
    MLPClassifier(activation="tanh", learning_rate="adaptive"),
    RandomForestClassifier(max_depth=11, n_estimators=20),
]

trainData = Data()
data = trainData.readTrainData()

trainData.predict(classifiers[2], 'AdaBoost', '')
