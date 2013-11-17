# Event
# Hackbright Data Science Workshop.

# Author
# Daniel Wiesenthal.  dw@cs.stanford.edu.

# What is this?
# This is a simple script illustrating the usage of the Python NLTK classifier.  It is written in Python, but the comments are intended to make it clear how to port to other languages.  The flow isn't particularly well decomposed as a program; rather, it is intended to go along linearly with the associated talk/presentation.
# The goal is to find out which chocolate a particular volunteer during the talk will like.  We have a few examples of chocolate bars that we know are either matches or not (misses), and want to use that to make a guess about an unknown bar (we don't know if it will be a match, and want to guess).

# Further reading:
# http://www.stanford.edu/class/cs124/lec/naivebayes.pdf
# http://nltk.googlecode.com/svn/trunk/doc/book/ch06.html

# Software Setup
# For this script to work, you'll need to have Python, NLTK (a Python package), and Numpy (upon which NLTK depends) installed.  On a Mac (which all have numpy pre-installed these days), run:
# sudo easy_install pip
# sudo pip install nltk
# <cd to directory with this file>
# python classification_101.py

# Build a classifier that does the training and the prediction for a given data set.
try:
    import nltk
    from nltk.classify.util import apply_features
    import string
    import math
    print "Great!  Looks like you're all set re: NLTK and Python."
except Exception, e:
    print "Bummer.  Looks like you don't have NLTK and Python set up correctly.  (Exception: "+str(e)+")"
    quit()

# training data set with known likes
known_1 = ("fruity dark organic sweet chocolate", "miss")
known_2 = ("interesting spicy dark bitter", "miss")
known_3 = ("sweet caramel crunchy light salty", "match")
known_4 = ("fruity dark organic bitter", "miss")
known_5 = ("sweet dark crunchy bitter interesting fruity", "match")
known_6 = ("light milky sweet", "match")
known_7 = ("refreshing dark sweet minty", "match")
known_8 = ("dark organic bitter", "miss")
known_9 = ("dark bitter bitter plain intense ghirardelli scary", "miss")
known_10 = ("organic dark salty bitter", "miss")

known_data_points = [known_1, known_2, known_3, known_4, known_5, known_6, known_7, known_8, known_9, known_10]

# sample data set for prediction testing
unknown_1 = "milky light sweet nutty"
unknown_2 = "dark bitter plain"
unknown_3 = "dark dark bitter beyond belief organic"
unknown_4 = "organic minty sweet dark"

def feature_extracting_function(data_point):
    features = {} #Dictionary, roughly equivalent to a hashtable in other languages.
    data_point = ''.join(ch for ch in data_point if ch not in set(string.punctuation)) #Strip punctuation characters from the string. In Python, this happens to be usually done with a .join on the string object, but don't be thrown if you're used to other languages and this looks weird (hell, it looks weird to me), all we're doing is stripping punctuation.
    words = data_point.split() #Split data_point on whitespace, return as list
    words = [word.lower() for word in words] #Convert all words in list to lowercase.  The [] syntax is a Python "list comprehension"; Google that phrase if you're confused.

    #Create a dictionary of features (True for each feature present, implicit False for absent features).  In this case, features are words, but they could be bigger or smaller, simpler or more complex.
    for word in words:
        features["contains_word_(%s)" % word] = True
    return features

train_set = apply_features(feature_extracting_function, known_data_points)

#Train a Naive Bayes Classifier (simple but surprisingly effective).  This isn't the only classifier one could use (dtree is another, and there are many, many more), but it's a good start.

# train set is a list of tuples containing hashes
# print train_set

# rewrite training data naive bayes classifier
# based on: prob = P(c) * P(F1 | c) * P(F2 | c) * P(F3 | c) ...
# Class is miss or match
# P(c = miss) = p(c)                           * p(organic|miss)
# P(c = miss) = count(miss)/total_count)       * count(organic AND miss)/count(miss)
def train_data(train_set):
    # use a nested hash to keep track of the occurrences in each category
    occurrences = {"miss": {}, "match": {}}
    counters    = {"miss": 0, "match": 0}
    for item in train_set:
        # item here is a tuple: ({'contains_word_(organic)' : True, ... "miss"})
        # each item in train_set is a tuple, and contains "miss" or "match" as the second
        # item in the tuple, and the feature data in the item. Lookup the key in the occurrences
        # hash, and add the features to it.
        counters[item[1]] += 1
        for key in item[0]:
            if key in occurrences[item[1]]:
                occurrences[item[1]][key] += 1
            else:
                occurrences[item[1]][key] = 1
            # print item[1], key, occurrences[item[1]][key]

    return occurrences, counters

trained, counter = train_data(train_set)

def predict_data(trained_data, counter, unknown_data):
    # print trained_data
    total_count   = float(counter['miss'] + counter['match'])
    prob_miss     = counter['miss'] / total_count
    prob_match    = counter['match'] / total_count

    # get the total number of misses and matches in our training data
    # for probability calculation
    # for key, value in trained_data['miss'].iteritems():
    #     total_misses += value
    # for key, value in trained_data['match'].iteritems():
    #     total_matches += value

    # features is a hash of present descriptors in our piece of data
    features = feature_extracting_function(unknown_data)

    # return a default value of 0.1 in order to handle zero values
    for item in features:

        prob_miss += math.log(trained_data['miss'].get(item, 0.1) / float(counter['miss']))
        prob_match += math.log(trained_data['match'].get(item, 0.1) / float(counter['match']))
    
    if prob_miss > prob_match:
        print prob_miss
        print prob_match
        return "Miss"
    else:
        print prob_miss
        print prob_match
        return "Match"

print predict_data(trained, counter, unknown_1)
print predict_data(trained, counter, unknown_2)
print predict_data(trained, counter, unknown_3)
print predict_data(trained, counter, unknown_4)