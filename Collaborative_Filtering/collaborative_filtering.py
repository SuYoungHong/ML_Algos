__author__ = 'coolguy aka Su-Young Hong'

import numpy as np
import csv

# your path to file here
training = 
testing = 
advertisers = 
outfile =  

data1 = np.loadtxt(training, delimiter= ',', skiprows= 1)
data2 = np.loadtxt(testing, delimiter= ',', skiprows= 1)

trainingUsers = set([str(i[0]) for i in data1])
testingUsers = set([str(i[0]) for i in data2])

data = np.concatenate((data1,data2))

userlist = {}
for i in data:
    if str(i[0]) in userlist:
        userlist[str(i[0])].append(str(i[1]))
    elif str(i[0]) not in userlist:
        userlist[str(i[0])] = [str(i[1])]

userSimilarity = {}
for i in userlist.keys():
    user1dict = {}
    for j in userlist.keys():
        user1 = set(userlist[i])
        common = user1.intersection(userlist[j])
        total = user1.union(userlist[j])
        if total != 0:
            similarity = len(common)/float(len(total))
        else:
            similarity = 0.0
        if i == j:
            pass
        else:
            if similarity == 0.0:
                pass
            else:
                user1dict[j] = similarity
    userSimilarity[i] = user1dict

recommendations = {}
for i in testingUsers:
    bestpick = {}
    for j in userSimilarity[i].keys():
        candidates = set(userlist[j]).difference(userlist[i])
        for k in candidates:
            if k in bestpick:
                bestpick[k] += userSimilarity[i][j]
            elif k not in bestpick:
                bestpick[k] = userSimilarity[i][j]
    recommend = max(bestpick, key = bestpick.get)
    recommendations[i] = recommend

advertDict = {}
with open(advertisers, 'r') as fp:
    fp.next()
    reader = csv.reader(fp)
    for i in reader:
        advertDict[str(float(i[0]))] = i[1]

results = []
for i in recommendations.keys():
    row = (int(float(i)), int(float(recommendations[i])), advertDict[recommendations[i]])
    results.append(row)

results.sort()

with open(outfile, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(results)

