import xml.etree.cElementTree as ET
import glob
import os
import re
# note importing Counter directly instead of as something to match with test_tfidf.py format
from collections import Counter
import numpy as np


# takes a path and returns list of paths that match that path
# if you put * somewhere in the path, that portion will be anything
# i.e. '/testing/*' will look for all files with the path the same
# up to the star, 'testing/'*.txt' will return all paths with the
# same format except the * which can be anything, so will return all
# text files inside testing/


def filelist(pathspec):
    files = glob.glob(pathspec)
    newfiles = []
    for i in files:
        if os.path.getsize(i) > 0:
            newfiles.append(i)
    return newfiles


# takes a path to an XML file and returns the text from <title> and
# <text>, including all <p> objects found within <text>

def get_text(fileName):
    tree = ET.ElementTree(file = fileName)
    # you can simplify this quite a bit. for example, there is only one title so there's no point in doing a loop around Roots
    # also, do a search like we did in class:
#    	for t in tree.iterfind('.//text//*'):

    Roots = tree.findall('.//title')
    text = ''
    for element in Roots:
        text = text + ' ' + element.text
    Roots = tree.findall('.//text')
    for element in Roots:
        text = text + ' ' + element.text
    Roots = tree.findall('.//p')
    for element in Roots:
        text = text + ' ' + element.text
    return text


# takes a long string, strips out all numbers, punctuation,
# tab, carriage return, newline with space, splits string with
# by space into list of words, remove all words smaller than 3 letter,
# make all words lower case, and return non-unique list of words

def words(doc):
    text = re.sub('[^A-Za-z]', ' ', doc)
    text = text.split()
    text = [i for i in text if len(i) > 2]
    text = [i.lower() for i in text]
    return text



# takes a list of files, like we would get from filelist
# and creates a tf_map and df: tf_map is a dictionary
# with each filepath (a string) mapped to that file's
# specific term frequency map (which itself is made up of
# every unique word in the file, mapped to that unique
# word's text frequency), so it's a map mapping files
# to their text frequency maps (which themselves are mapping
# words to that word's text frequency). the df is the
# document frequency of all the words in all the files
def create_indexes(listoffiles):
    df = Counter()
    tf_map = {}
    for f in listoffiles:
        d = get_text(f)
        text = words(d)
        n = float(len(text))
        tf = Counter(text)
        for t in tf:
            tf[t] = tf[t]/n
            df[t] += 1
        tf_map[f] = tf
    return (tf_map, df)

# takes a single tf (which is a single file's map of all
# it's terms mapped to all of those term's frequencies...)
# and converts that map into a tfidf map, which is a map
# of each term's tfidf score. Requires a document frequency
# df of the entire corpus (all the files) and N, the number
# of files/documents... note, this is an in between function
# we just insert it into the create_tfidif_map function

def doc_tfidf(tf, df, N):
    tfidf = {}
    N = float(N)
    for t in tf:
        dft = df[t]/N
        idft = 1/dft
        tfidf[t] = tf[t] * np.log(idft)
    return tfidf


# takes a list of files and creates a tfidf map (so a dicitonary
# mapping each file to a dictionary of each word in the file
# mapped to it's tfidf score), requires the create_indexes
# function above to be used with the files, and the
# doc_tfidif function in the loop, to be used with each
# document's individual tf map from the result of create_indexes

def create_tfidf_map(files):
    (tf_map, df) = create_indexes(files)
    tfidf_map = {}
    N = len(files)
    for f in files:
        tfidf = doc_tfidf(tf_map[f], df, N)
        tfidf_map[f] = tfidf
    return tfidf_map
