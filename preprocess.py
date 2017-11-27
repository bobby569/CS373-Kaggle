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
