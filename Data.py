from preprocess import *
import numpy as np
import csv

class Data:

    def __init__(self):
        self.data = []
        self.attr = None
        self.target = None

    def readTrainData(self):
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
                getDecimal(row['Amount Requested']) / 100,
                getDecimal(row['Amount Funded By Investors']) / 100,
                getPercentage(row['Interest Rate']),
                getLoanLen(row['Loan Length']),
                getCreditGrade(row['CREDIT Grade']),
                getStringHash(row['Loan Purpose']),
                getDecimal(row['Monthly PAYMENT']),
                getDecimal(row['Total Amount Funded']) / 100,
                getPercentage(row['Debt-To-Income Ratio']),
                getStringHash(row['Home Ownership']) % 11,
                getDecimal(row['Monthly Income']),
                getFICOScore(row['FICO Range']),
                getCreditTime(row['Earliest CREDIT Line']),
                getDecimal(row['Open CREDIT Lines']),
                getDecimal(row['Total CREDIT Lines']),
                getDecimal(row['Revolving CREDIT Balance']),
                getPercentage(row['Revolving Line Utilization'], 0.),
                getDecimal(row['Inquiries in the Last 6 Months'], -1),
                getDecimal(row['Accounts Now Delinquent'], -1),
                getDecimal(row['Months Since Last Delinquency'], -1),
                getDecimal(row['Public Records On File'], -1),
                getEducation(row['Education']),
                getEmployeement(row['Employment Length']),
                int(row['Status (Fully Paid=1, Not Paid=0)'])
            ])
            self.data.append(obj)
        trainfile.close()

        d = np.array(self.data)
        self.attr = d[:, :-1]
        self.target = map(int, d[:, -1])

        return d

    def predict(self, classifier, name, para, i='0'):
        classifier.fit(self.attr, self.target)

        toPredictFile = open('./Loan_ToPredict.csv')
        reader = csv.DictReader(toPredictFile)
        outputfile = open('./%s_%s.csv' % (name, i), 'w')
        writer = csv.writer(outputfile, delimiter=',')
        writer.writerow(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'])

        for row in reader:
            test = np.array([[
                getDecimal(row['Amount Requested']) / 100,
                getDecimal(row['Amount Funded By Investors']) / 100,
                getPercentage(row['Interest Rate']),
                getLoanLen(row['Loan Length']),
                getCreditGrade(row['CREDIT Grade']),
                getStringHash(row['Loan Purpose']),
                getDecimal(row['Monthly PAYMENT']),
                getDecimal(row['Total Amount Funded']) / 100,
                getPercentage(row['Debt-To-Income Ratio']),
                getStringHash(row['Home Ownership']) % 11,
                getDecimal(row['Monthly Income']),
                getFICOScore(row['FICO Range']),
                getCreditTime(row['Earliest CREDIT Line']),
                getDecimal(row['Open CREDIT Lines']),
                getDecimal(row['Total CREDIT Lines']),
                getDecimal(row['Revolving CREDIT Balance']),
                getPercentage(row['Revolving Line Utilization'], 0.),
                getDecimal(row['Inquiries in the Last 6 Months'], -1),
                getDecimal(row['Accounts Now Delinquent'], -1),
                getDecimal(row['Months Since Last Delinquency'], -1),
                getDecimal(row['Public Records On File'], -1),
                getEducation(row['Education']),
                getEmployeement(row['Employment Length']),
            ]])
            res = classifier.predict(test)[0] * para
            writer.writerow([row['Loan ID'], res])

        outputfile.close()
        toPredictFile.close()
