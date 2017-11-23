import numpy as np
import os
import csv

path = "./"
files = os.listdir(path)
filepath = [path + f for f in files if ".csv" in f]

fileptr = map(open, filepath)
for f in fileptr:
    f.readline()

weights = np.array([0.802, 0.780, 0.796, 0.823, 0.777])
threhold = np.sum(weights) / 2;

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
