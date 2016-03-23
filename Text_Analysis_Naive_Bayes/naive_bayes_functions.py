import glob
import os
import random
import re
import collections
import numpy as np


# takes x, a path to a directory. Directory should have subdirectories, containing all the
# txt files of a given class. So all docs need to be .txt, and all docs of same class need
# to be together in same subdirectory, directly under x.
# returns a Cidentity, a dictionary of classes and their associated files, as well as
# the entire corpus divided into 2/3 (a training set) and 1/3 (a test set) who are non-overlapping
# the sample set can be changed if desired...
def sortnsplit(x):
    Clist = os.listdir(x)
    Cidentity = {}
    Master = []
    for i in Clist:
        path = x + '/' + i
        if os.path.isdir(path):
            lfiles = glob.glob(path + '/*.txt')
            Cidentity[i] = lfiles
            for j in lfiles:
                Master.append(j)
    group1 = random.sample(Master, ((len(Master))/3))
    remaining = set(Master).difference(group1)
    group2 = random.sample(remaining, ((len(Master))/3))
    group3 = set(remaining).difference(group2)
    group3 = list(group3)
    return (Cidentity, group1, group2, group3)


# takes a list of filepaths and re-sorts them according to the classes
# returns x lists, note, 'good' and 'bad' are hardcoded and this code
# assumes two classes, good and bad
def assignsample(x, identitydict):
    positive = set(identitydict['pos']).intersection(x)
    positive = list(positive)
    negative = set(identitydict['neg']).intersection(x)
    negative = list(negative)
    return (positive, negative)


# takes a list of filepaths and turns the content in those files into
# a BAG OF WORDS!
def bagofwords(x):
    text = ''
    for path in x:
        a = open(path, 'r')
        text = text + ' ' + a.read()
        a.close()
    text = re.sub('[^A-Za-z\s]', '', text)
    text = text.lower()
    text = text.split()
    return text


# takes two lists, each a list of paths for a particular class
# returns a map containing the conditional probabilities of each word
# in each class, and the document probability of each class
# note, pos and neg are hardcoded, so the positive set NEEDS to be
# the first arg, neg the second
# note, i changed the structure so that it returns a dictionary with
# each class as the keys, and inside each class's dictionary is a
# dictionary, with p(c) inside the key 'pc' and the Counter object
# with every word's p(w|c) inside the key pwc
def trainingdata(pos, neg):
    trained = {}
    # so P(c), the probability of the class to the total group of training docs
    trained['pos'] = {}
    trained['neg'] = {}
    trained['pos']['pc'] = len(pos)/float(len(pos) + len(neg))
    trained['neg']['pc'] = len(neg)/float(len(pos) + len(neg))
    posbag = bagofwords(pos)
    negbag = bagofwords(neg)
    poswordcount = float(len(posbag))
    negwordcount = float(len(negbag))
    # so this creates a frequency map of each word for each class...
    # right now, it's NOT in probability terms, just the number of each
    # word in each class
    trained['pos']['pwc'] = collections.Counter(posbag)
    trained['neg']['pwc'] = collections.Counter(negbag)
    # V is the number of unique words in the entire superset
    V = len(set(trained['pos']['pwc']).union(set(trained['neg']['pwc'])))
    # create iterkeys just so we can do stuff faster
    positer1 = trained['pos']['pwc'].iterkeys()
    negiter1 = trained['neg']['pwc'].iterkeys()
    # convert the COUNT of each word in the wordfreqs to a conditional probability,
    # p(w|c) = (count(w,c) + 1) / (count(c) + |v| + 1)
    for i in positer1:
        trained['pos']['pwc'][i] = (trained['pos']['pwc'][i] + 1)/(poswordcount + V + 1)
    for i in negiter1:
        trained['neg']['pwc'][i] = (trained['neg']['pwc'][i] + 1)/(negwordcount + V + 1)
    # adding my UNK probabilities now so I can reference it later
    # in the actual check class function... note, I labeled it 'UNK0' because I know
    # no words in the corpus will be called this because I already stripped out all numbers
    # earlier in the bag of words
    trained['pos']['pwc']['UNK0'] = 1/(poswordcount + V + 1)
    trained['neg']['pwc']['UNK0'] = 1/(negwordcount + V + 1)
    return trained

# just wrote some code because I wanted to test out the naive bayes and conditional
# probability calculation theoretically, below code is not needed for the function
# def iwantknow(bag1, bag2):
#     lenbag1 = len(bag1)
#     lenbag2 = len(bag2)
#     count1 = set(collections.Counter(bag1))
#     count2 = set(collections.Counter(bag2))
#     uniques = len(count1.union(count2))
#     cushion1 = (len(count1.difference(count2)) + 1)/ float(lenbag2 + 1 + uniques)
#     cushion2 = (len(count2.difference(count1)) + 1)/ float(lenbag1 + 1 + uniques)
#     UNK1 = (1)/ float(lenbag2 + 1 + uniques)
#     UNK2 = (1)/ float(lenbag1 + 1 + uniques)
#     return cushion1, cushion2, UNK1, UNK2, uniques



# run a naive bayes algo on a new document, needs existing trained data, and
# will compare the naive bayes calculation for each class and return the class
# most likely
def naivebayes(path, trainedata):
    # note, I'm defining this function to look at one document at a time, but
    # since bag of words expects a list of paths, I just put the one path into
    # a list of one element
    words = bagofwords([path])
    words = collections.Counter(words)
    bayes = {}
    for i in trainedata.keys():
        score = 0
        score += np.log(trainedata[i]['pc'])
        for j in words:
            pwc = trainedata[i]['pwc'][j]
            if pwc == 0:
                pwc = trainedata[i]['pwc']['UNK0']
            score += words[j] * np.log(pwc)
        bayes[score] = i
    return bayes[max(bayes)]

# this function takes a list of paths, the testing set, and uses
# the bayes method with naivebayes function on each path, and
# returns a dictionary of classes with the filepaths of all the
# documents that are predicted to be in that path
# note, I've hardcoded pos and neg, so this function is specific to
# this homework
def assigntestgroup(listpaths, traindata):
    result = {'pos':[],'neg':[]}
    for i in listpaths:
        guess = naivebayes(i, traindata)
        result[guess].append(i)
    return result

# this takes the results of the assigntestgroup and return a
# percentage of accuracy, by checking each item in the assigned
# class and checking against the masterkey, which we get from sort
# n split function above




def homeworkoutput(master, training, testgroup):
    trainpos, trainneg = assignsample(training, master)
    trainedmap = trainingdata(trainpos, trainneg)
    testing = assigntestgroup(testgroup, trainedmap)
    numtrainpos = len(trainpos)
    numtrainneg = len(trainneg)
    testpos, testneg = assignsample(testgroup, master)
    numtestpos = len(testpos)
    numtestneg = len(testneg)
    poscor, posincor = assignsample(testing['pos'], master)
    negincor, negcor = assignsample(testing['neg'], master)
    numposcorrect = len(poscor)
    numnegcorrect = len(negcor)
    print 'num_pos_test_docs:', numtestpos
    print 'num_pos_training_docs:', numtrainpos
    print 'num_pos_correct_docs:', numposcorrect
    print 'num_neg_test_docs:', numtestneg
    print 'num_neg_training_docs:', numtrainneg
    print 'num_neg_correct_docs:', numnegcorrect
    accuracy = round((((numposcorrect + numnegcorrect)/float(len(testgroup)))*100),2)
    print 'accuracy:', accuracy, '%'
    return accuracy