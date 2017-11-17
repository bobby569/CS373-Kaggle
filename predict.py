from Data import Data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
                            BaggingClassifier, ExtraTreesClassifier, \
                            GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier

classifiers = [
    DecisionTreeClassifier(max_depth=10, criterion="entropy"),
    AdaBoostClassifier(learning_rate=1.2),
    RandomForestClassifier(max_depth=10, n_estimators=30,
                           warm_start=True, bootstrap=False,
                           criterion="entropy"),
    BaggingClassifier(n_estimators=40, warm_start=True),
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=3, warm_start=True)
]

trainData = Data()
data = trainData.readTrainData()

trainData.predict(classifiers[4], 'GradientBoosting', '0.4_110')
