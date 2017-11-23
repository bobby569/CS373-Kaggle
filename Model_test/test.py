from Data import Data
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
                            BaggingClassifier, ExtraTreesClassifier, \
                            GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

classifiers = [
    DecisionTreeClassifier(max_depth=10, criterion="entropy"),      # 0.802
    AdaBoostClassifier(learning_rate=1.2),                          # 0.780
    RandomForestClassifier(max_depth=10, n_estimators=30,
                           bootstrap=True, criterion="entropy"),    # 0.796
    BaggingClassifier(n_estimators=35),                             # 0.823
    ExtraTreesClassifier(max_depth=25, n_estimators=40,
                         criterion="entropy"),                      # 0.777
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=3),                        # 0.841
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=4),                        # 0.841
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=5),                        # 0.835
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=6),                        # 0.832
    RidgeClassifier(alpha=2),                                       # 0.728
]

d = Data()
d.readTrainData()

c = classifiers[9]
iteration = 5

total = 0
for _ in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.3)
    c.fit(X_train, y_train)
    score = c.score(X_test, y_test)
    print score
    total += score

print total / iteration
