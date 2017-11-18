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
            obj = np.array([
                getDecimal(row['Amount Requested']) // 100,
                getDecimal(row['Amount Funded By Investors']) // 100,
                getPercentage(row['Interest Rate']),
                # getDecimal(row['Amount Funded By Investors']) * getPercentage(row['Interest Rate']) // 10000,
                getLoanLen(row['Loan Length']),
                getCreditGrade(row['CREDIT Grade']),
                getStringHash(row['Loan Purpose']),
                getDecimal(row['Monthly PAYMENT']) // 1,
                getDecimal(row['Total Amount Funded']) // 100,
                getPercentage(row['Debt-To-Income Ratio']) * 10 // 1 / 10,
                getStringHash(row['Home Ownership']),
                getDecimal(row['Monthly Income']) // 100,
                getFICOScore(row['FICO Range']),
                # getCreditTime(row['Earliest CREDIT Line']),
                # getDecimal(row['Total CREDIT Lines']),
                # getDecimal(row['Open CREDIT Lines']),
                getRatio(row['Total CREDIT Lines'], row['Open CREDIT Lines']),
                getDecimal(row['Revolving CREDIT Balance']) / 1000,
                getPercentage(row['Revolving Line Utilization'], 0.),
                getDecimal(row['Inquiries in the Last 6 Months'], -1),
                getDecimal(row['Accounts Now Delinquent'], -1),
                # getDecimal(row['Delinquent Amount'], -1),
                # getDecimal(row['Delinquencies (Last 2 yrs)'], -1),
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

    def predict(self, classifier, name, i='0'):
        classifier.fit(self.attr, self.target)

        toPredictFile = open('./Loan_ToPredict.csv')
        reader = csv.DictReader(toPredictFile)
        outputfile = open('./%s_%s.csv' % (name, i), 'w')
        writer = csv.writer(outputfile, delimiter=',')
        writer.writerow(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'])

        for row in reader:
            test = np.array([[
                getDecimal(row['Amount Requested']) // 100,
                getDecimal(row['Amount Funded By Investors']) // 100,
                getPercentage(row['Interest Rate']),
                # getDecimal(row['Amount Funded By Investors']) * getPercentage(row['Interest Rate']) // 10000,
                getLoanLen(row['Loan Length']),
                getCreditGrade(row['CREDIT Grade']),
                getStringHash(row['Loan Purpose']),
                getDecimal(row['Monthly PAYMENT']) // 1,
                getDecimal(row['Total Amount Funded']) // 100,
                getPercentage(row['Debt-To-Income Ratio']) * 10 // 1 / 10,
                getStringHash(row['Home Ownership']),
                getDecimal(row['Monthly Income']) // 100,
                getFICOScore(row['FICO Range']),
                # getCreditTime(row['Earliest CREDIT Line']),
                # getDecimal(row['Total CREDIT Lines']),
                # getDecimal(row['Open CREDIT Lines']),
                getRatio(row['Total CREDIT Lines'], row['Open CREDIT Lines']),
                getDecimal(row['Revolving CREDIT Balance']) / 1000,
                getPercentage(row['Revolving Line Utilization'], 0.),
                getDecimal(row['Inquiries in the Last 6 Months'], -1),
                getDecimal(row['Accounts Now Delinquent'], -1),
                # getDecimal(row['Delinquent Amount'], -1),
                # getDecimal(row['Delinquencies (Last 2 yrs)'], -1),
                getDecimal(row['Months Since Last Delinquency'], -1),
                getDecimal(row['Public Records On File'], -1),
                getEducation(row['Education']),
                getEmployeement(row['Employment Length']),
            ]])
            res = classifier.predict(test)[0]
            writer.writerow([row['Loan ID'], res])

        outputfile.close()
        toPredictFile.close()
