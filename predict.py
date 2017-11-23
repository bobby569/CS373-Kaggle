from Data import Data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
                            BaggingClassifier, ExtraTreesClassifier, \
                            GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

classifiers = [
    DecisionTreeClassifier(max_depth=10, criterion="entropy"),      # 0.802
    RandomForestClassifier(max_depth=10, n_estimators=30,
                           bootstrap=True, criterion="entropy"),    # 0.796
    BaggingClassifier(n_estimators=35),                             # 0.823
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=3),                        # 0.841
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=4),                        # 0.841
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=5),                        # 0.835
    GradientBoostingClassifier(learning_rate=0.4, n_estimators=110,
                               max_depth=6),                        # 0.832
]

trainData = Data()
data = trainData.readTrainData()

trainData.predict(classifier=classifiers[5], name='ExtraTrees', para=0.841, i='3')
