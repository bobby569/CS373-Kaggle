from Data import Data
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
                             GradientBoostingClassifier, VotingClassifier
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
]

c = VotingClassifier(
    estimators=classifier,
    voting='soft',
    weights=[0.8319,0.8337,0.8457,0.8464]
)

trainData = Data()
data = trainData.readTrainData()

trainData.predict(classifier=c, name='Voting', para=1, i='3')
