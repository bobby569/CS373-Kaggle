from Data import Data
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
                            BaggingClassifier, ExtraTreesClassifier, \
                            GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier

classifiers = [
    DecisionTreeClassifier(max_depth=10, criterion="entropy"),      # 0.80
    ExtraTreeClassifier(max_depth=13, criterion="entropy"),         # 0.69
    BernoulliNB(alpha=5.5),                                         # 0.65
    AdaBoostClassifier(learning_rate=1.2),                          # 0.79
    RandomForestClassifier(max_depth=10, n_estimators=30,
                           warm_start=True, bootstrap=False,
                           criterion="entropy"),                    # 0.83
    BaggingClassifier(n_estimators=40, warm_start=True),            # 0.95
    ExtraTreesClassifier(max_depth=25, n_estimators=40,
                         criterion="entropy"),                      # 0.78
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=3, warm_start=True),       # 0.86
    RidgeClassifier(alpha=2)                                        # 0.73
]

d = Data()
d.readTrainData()

c = classifiers[9]
iteration = 5

total = 0
for _ in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.2)
    c.fit(X_train, y_train)
    total += c.score(X_test, y_test)

print total / iteration
