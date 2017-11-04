import os
import csv

path = "./result/"
files = os.listdir(path)
filepath = [path + f for f in files if ".csv" in f]

fileptr = map(open, filepath)
for f in fileptr:
    f.readline()

divisor = len(fileptr) / 2 + 1

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
            val += int(s[1])
        writer.writerow([lid, val / divisor])
    except:
        break


outputfile.close()
for fptr in fileptr:
    fptr.close()
