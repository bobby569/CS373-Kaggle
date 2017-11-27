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
                getInteger(row['Amount Requested']) / 100,
                getInteger(row['Amount Funded By Investors']) / 100,
                getPercentage(row['Interest Rate']),
                getLoanLen(row['Loan Length']),
                getCreditGrade(row['CREDIT Grade']),
                getStringHash(row['Loan Purpose']),
                getInteger(row['Monthly PAYMENT']),
                getInteger(row['Total Amount Funded']) / 100,
                getPercentage(row['Debt-To-Income Ratio']),
                getStringHash(row['Home Ownership']) % 11,
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
                int(row['Status (Fully Paid=1, Not Paid=0)'])
            ])
            self.data.append(obj)
        trainfile.close()
        d = np.array(self.data)
        self.attr = d[:, :-1]
        self.target = map(int, d[:, -1])

        return d
