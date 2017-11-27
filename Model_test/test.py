from Data import Data
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
                            BaggingClassifier, ExtraTreesClassifier, \
                            GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import VotingClassifier

clf1 = DecisionTreeClassifier(max_depth=10, criterion="entropy")
clf2 = GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                           max_depth=4)
clf3 = BaggingClassifier(n_estimators=35)
clf4 = GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                           max_depth=3)
clf5 = GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                           max_depth=5)
clf6 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7, criterion="entropy"),
                   n_estimators=10, learning_rate=0.3)

d = Data()
d.readTrainData()

c = VotingClassifier(
    estimators=[('dt', clf1), ('b', clf3), ('gb4', clf4), ('gb2', clf2), ('gb5', clf5), ('ab', clf6)],
    voting='soft'
)
iteration = 1

total = 0
for _ in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.3)
    c.fit(X_train, y_train)
    score = c.score(X_test, y_test)
    total += score

print "%.4f" % (total / iteration)
