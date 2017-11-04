from Data import Data
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
                            BaggingClassifier, ExtraTreesClassifier, \
                            GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier


classifiers = [
    DecisionTreeClassifier(max_depth=10),    # 0
    GaussianNB(),
    AdaBoostClassifier(),   # 2
    MLPClassifier(),
    RandomForestClassifier(max_depth=10, n_estimators=20),
]

d = Data()
d.readTrainData()

c = classifiers[-1]

total = 0
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.2)
    c.fit(X_train, y_train)
    total += c.score(X_test, y_test)

print total
