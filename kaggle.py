from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
                             ExtraTreesClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv
import datetime

def getInteger(s, default=0):
    return int(float(s)) if s else default

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
    )
]

clf = VotingClassifier(
    estimators=classifiers,
    voting='soft'
)

data = []
trainfile = open('./Loan_Training.csv')
reader = csv.DictReader(trainfile)

for row in reader:
    if getInteger(row['Monthly Income']) > 200000 or \
        getCreditTime(row['Earliest CREDIT Line']) < 0 or \
        getInteger(row['Open CREDIT Lines']) > 40 or \
        getInteger(row['Total CREDIT Lines']) > 84 or \
        getInteger(row['Revolving CREDIT Balance']) > 700000 or \
        getPercentage(row['Revolving Line Utilization'], 0.) > 110 or \
        getInteger(row['Inquiries in the Last 6 Months'], -1) > 22 or \
        getInteger(row['Delinquent Amount'], -1) > 1000:
        continue

    obj = np.array([
        getInteger(row['Amount Requested']),
        getInteger(row['Amount Funded By Investors']),
        getPercentage(row['Interest Rate']),
        getLoanLen(row['Loan Length']),
        getCreditGrade(row['CREDIT Grade']),
        getInteger(row['Monthly PAYMENT']),
        getInteger(row['Total Amount Funded']),
        getPercentage(row['Debt-To-Income Ratio']),
        getInteger(row['Monthly Income']),
        getFICOScore(row['FICO Range']),
        getCreditTime(row['Earliest CREDIT Line']),
        getInteger(row['Open CREDIT Lines']),
        getInteger(row['Total CREDIT Lines']),
        getInteger(row['Revolving CREDIT Balance']),
        getPercentage(row['Revolving Line Utilization'], 0.),
        getInteger(row['Inquiries in the Last 6 Months'], -1),
        getInteger(row['Accounts Now Delinquent'], -1),
        getInteger(row['Delinquent Amount'], -1),
        getInteger(row['Months Since Last Delinquency'], -1),
        getInteger(row['Public Records On File'], -1),
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
        getInteger(row['Amount Requested']),
        getInteger(row['Amount Funded By Investors']),
        getPercentage(row['Interest Rate']),
        getLoanLen(row['Loan Length']),
        getCreditGrade(row['CREDIT Grade']),
        getInteger(row['Monthly PAYMENT']),
        getInteger(row['Total Amount Funded']),
        getPercentage(row['Debt-To-Income Ratio']),
        getInteger(row['Monthly Income']),
        getFICOScore(row['FICO Range']),
        getCreditTime(row['Earliest CREDIT Line']),
        getInteger(row['Open CREDIT Lines']),
        getInteger(row['Total CREDIT Lines']),
        getInteger(row['Revolving CREDIT Balance']),
        getPercentage(row['Revolving Line Utilization'], 0.),
        getInteger(row['Inquiries in the Last 6 Months'], -1),
        getInteger(row['Accounts Now Delinquent'], -1),
        getInteger(row['Delinquent Amount'], -1),
        getInteger(row['Months Since Last Delinquency'], -1),
        getInteger(row['Public Records On File'], -1),
        getEducation(row['Education']),
        getEmployeement(row['Employment Length']),
    ]])
    res = clf.predict(test)[0]
    writer.writerow([row['Loan ID'], res])

outputfile.close()
toPredictFile.close()
