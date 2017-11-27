from Data import Data
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

c = VotingClassifier(
    estimators=[('dt', clf1), ('b', clf3), ('gb4', clf4), ('gb2', clf2), ('gb5', clf5), ('ab', clf6)],
    voting='soft'
)

trainData = Data()
data = trainData.readTrainData()

trainData.predict(classifier=c, name='Voting', para=1, i='1')
