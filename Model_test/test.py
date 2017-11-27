from Data import Data
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
                             ExtraTreesClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, \
                                 RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

classifier = [
    (
        'ab',
         AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=7, criterion="entropy"),
            n_estimators=10, learning_rate=0.3)
    ),
    (
        'b',
        BaggingClassifier(n_estimators=35)
    ),
    (
        'et',
        ExtraTreeClassifier()
    ),
    (
        'gb3',
        GradientBoostingClassifier(learning_rate=0.4, n_estimators=110, max_depth=3)
    ),
    (
        'gb4',
        GradientBoostingClassifier(learning_rate=0.4, n_estimators=110, max_depth=4)
    ),
    (
        'gb5',
        GradientBoostingClassifier(learning_rate=0.4, n_estimators=110, max_depth=5)
    ),
    (
        'dt',
        DecisionTreeClassifier(max_depth=10, criterion="entropy")
    )
]

d = Data()
d.readTrainData()

c = VotingClassifier(
    estimators=classifier,
    voting='soft'
)
# c = classifier[3][1]
iteration = 3

total = 0
for _ in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.3, random_state=42)
    c.fit(X_train, y_train)
    score = c.score(X_test, y_test)
    total += score

print "%.4f" % (total / iteration)
