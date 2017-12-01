from Data import Data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
                             ExtraTreesClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

classifier = [
    (
        'ab',
         AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=7, criterion="entropy"),
            n_estimators=10, learning_rate=0.25)
    ),      # 0.8319
    (
        'b',
        BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=20, criterion="entropy"),
            n_estimators=35)
    ),      # 0.8337
    (
        'gb3',
        GradientBoostingClassifier(learning_rate=0.4, n_estimators=110, max_depth=3)
    ),      # 0.8457
    (
        'gb4',
        GradientBoostingClassifier(learning_rate=0.4, n_estimators=110, max_depth=4)
    ),      # 0.8464
    (
        'gb5',
        GradientBoostingClassifier(learning_rate=0.4, n_estimators=110, max_depth=5)
    )       # 0.8376
]

d = Data()
d.readTrainData()

c = VotingClassifier(
    estimators=classifier,
    voting='soft',
    weights=[0.8319,0.8337,0.8457,0.8464,0.8376]
)

iteration = 2

total = 0
for _ in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(d.attr, d.target, test_size=0.3)
    c.fit(X_train, y_train)
    score = c.score(X_test, y_test)
    print score
    total += score

print "%.4f" % (total / iteration)
