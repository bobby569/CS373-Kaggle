import numpy as np
import os
import csv

path = "./"
files = os.listdir(path)
filepath = [path + f for f in files if ".csv" in f][:-1]
if './result.csv' in filepath:
    filepath.remove('./result.csv')

weights = np.array([0.] * len(filepath))
for i in xrange(len(filepath)):
    weights[i] = int(filepath[i][2:7]) / 100000.0
threhold = np.sum(weights) / 2;

# print np.sum(weights) / len(weights)
#
# exit()

fileptr = map(open, filepath)
for f in fileptr:
    f.readline()

outputfile = open("./result.csv", "w")
writer = csv.writer(outputfile, delimiter=',')
writer.writerow(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'])

while True:
    try:
        lid = ""
        val = 0
        for f in fileptr:
            s = f.readline().split(',')
            lid = s[0]
            val += float(s[1])
        writer.writerow([lid, 1 if val > threhold else 0])
    except:
        break


outputfile.close()
for fptr in fileptr:
    fptr.close()
