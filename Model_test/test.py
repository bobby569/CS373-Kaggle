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
    DecisionTreeClassifier(max_depth=10, criterion="entropy"),      # 0.808
    AdaBoostClassifier(learning_rate=1.2),                          # 0.786
    RandomForestClassifier(max_depth=10, n_estimators=30,
                           bootstrap=True, criterion="entropy"),    # 0.798
    BaggingClassifier(n_estimators=35),                             # 0.830
    ExtraTreesClassifier(max_depth=25, n_estimators=40,
                         criterion="entropy"),                      # 0.782
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=4, warm_start=True),       # 0.960
    RidgeClassifier(alpha=2)                                        # 0.733
]

d = Data()
d.readTrainData()

c = classifiers[5]
iteration = 5

total = 0
for _ in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.2)
    c.fit(X_train, y_train)
    total += c.score(X_test, y_test)

print total / iteration
