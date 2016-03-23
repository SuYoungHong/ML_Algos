from naive_bayes_functions import *
import argparse

def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

def main():
    args = parseArgument()
    directory = args['d'][0]
    print directory
    master, one, two, three = sortnsplit(directory)
    print 'iteration 1'
    firstresult = homeworkoutput(master, one + two, three)
    print 'iteration 2'
    secondresult = homeworkoutput(master, one + three, two)
    print 'iteration 3'
    thirdresult = homeworkoutput(master, two + three, one)
    final = (firstresult + secondresult + thirdresult)/3
    print 'ave_accuracy:', final, '%'

main()