import datetime

def getDecimal(s, default=0.):
    return float(s) if s else default

def getPercentage(s, default=0.):
    return float(s[:-1]) if s else default

def getLoanLen(s):
    if "36" in s:
        return 1
    elif "60" in s:
        return 2
    else:
        return 0

def getCreditGrade(s):
    return 35 - (ord(s[0]) - 65) * 5 - int(s[1])

def getStringHash(s):
    return hash(s) % 100000007

def getFICOScore(s):
    return (int(s[:3]) - 640) / 5 + 1 if s else 0

def getCreditTime(s):
    try:
        t = datetime.datetime.strptime(s, '%m/%d/%y')
        res = (t.year - 1950) * 12
        res += (t.month - 12)
        return res
    except:
        return 0

def getRatio(nu, de):
    nu = getDecimal(nu) + 0.1
    de = getDecimal(de) + 0.1
    return nu / de  * 10 // 1 / 10

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
