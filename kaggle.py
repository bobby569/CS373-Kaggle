from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
                             ExtraTreesClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv
import datetime

def getDecimal(s, default=0):
    return float(s) if s else default

def getPercentage(s, default=0.):
    return float(s[:-1]) if s else default

def getLoanLen(s):
    return 1 if "60" in s else 0

def getCreditGrade(s):
    return 36 - (ord(s[0]) - 65) * 5 - int(s[1])

def getStringHash(s):
    return hash(s) % 1007 % 100

def getFICOScore(s):
    return (int(s[:3]) - 640) / 5 + 1 if s else 0

def getCreditTime(s):
    try:
        t = datetime.datetime.strptime(s, '%m/%d/%y')
        res = (t.year - 1961) * 12
        res += (t.month - 12)
        return res
    except:
        return 0

def getEducation(s):
    return 1 if s else 0

def getEmployeement(s):
    if "<" in s:
        return 0
    if "+" in s:
        return 10
    try:
        return int(s.split()[0])
    except:
        return -1


classifiers = [
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

clf = VotingClassifier(
    estimators=classifiers,
    voting='soft',
    weights=[0.8319,0.8337,0.8457,0.8464,0.8376]
)

data = []
trainfile = open('./Loan_Training.csv')
reader = csv.DictReader(trainfile)

for row in reader:
    if getDecimal(row['Monthly Income']) > 200000 or \
        getCreditTime(row['Earliest CREDIT Line']) < 0 or \
        getDecimal(row['Open CREDIT Lines']) > 40 or \
        getDecimal(row['Total CREDIT Lines']) > 84 or \
        getDecimal(row['Revolving CREDIT Balance']) > 700000 or \
        getPercentage(row['Revolving Line Utilization'], 0.) > 110 or \
        getDecimal(row['Inquiries in the Last 6 Months'], -1) > 22:
        continue

    obj = np.array([
        getDecimal(row['Amount Requested']),
        getDecimal(row['Amount Funded By Investors']),
        getPercentage(row['Interest Rate']),
        getLoanLen(row['Loan Length']),
        getCreditGrade(row['CREDIT Grade']),
        getDecimal(row['Monthly PAYMENT']),
        getDecimal(row['Total Amount Funded']),
        getPercentage(row['Debt-To-Income Ratio']),
        getDecimal(row['Monthly Income']),
        getFICOScore(row['FICO Range']),
        getCreditTime(row['Earliest CREDIT Line']),
        getDecimal(row['Open CREDIT Lines']),
        getDecimal(row['Total CREDIT Lines']),
        getDecimal(row['Revolving CREDIT Balance']),
        # getPercentage(row['Revolving Line Utilization'], 0.),
        getDecimal(row['Inquiries in the Last 6 Months'], -1),
        getDecimal(row['Accounts Now Delinquent'], -1),
        getDecimal(row['Months Since Last Delinquency'], -1),
        # getDecimal(row['Public Records On File'], -1),
        getEducation(row['Education']),
        getEmployeement(row['Employment Length']),
        row['Status (Fully Paid=1, Not Paid=0)']
    ])
    data.append(obj)
trainfile.close()

data = np.array(data)
train_X = data[:, :-1]
train_y = map(int, data[:, -1])

clf.fit(train_X, train_y)

toPredictFile = open('./Loan_ToPredict.csv')
reader = csv.DictReader(toPredictFile)
outputfile = open('./result.csv', 'w')
writer = csv.writer(outputfile, delimiter=',')
writer.writerow(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'])

for row in reader:
    test = np.array([[
        getDecimal(row['Amount Requested']),
        getDecimal(row['Amount Funded By Investors']),
        getPercentage(row['Interest Rate']),
        getLoanLen(row['Loan Length']),
        getCreditGrade(row['CREDIT Grade']),
        getDecimal(row['Monthly PAYMENT']),
        getDecimal(row['Total Amount Funded']),
        getPercentage(row['Debt-To-Income Ratio']),
        getDecimal(row['Monthly Income']),
        getFICOScore(row['FICO Range']),
        getCreditTime(row['Earliest CREDIT Line']),
        getDecimal(row['Open CREDIT Lines']),
        getDecimal(row['Total CREDIT Lines']),
        getDecimal(row['Revolving CREDIT Balance']),
        # getPercentage(row['Revolving Line Utilization'], 0.),
        getDecimal(row['Inquiries in the Last 6 Months'], -1),
        getDecimal(row['Accounts Now Delinquent'], -1),
        getDecimal(row['Months Since Last Delinquency'], -1),
        # getDecimal(row['Public Records On File'], -1),
        getEducation(row['Education']),
        getEmployeement(row['Employment Length']),
    ]])
    res = clf.predict(test)[0]
    writer.writerow([row['Loan ID'], res])

outputfile.close()
toPredictFile.close()
