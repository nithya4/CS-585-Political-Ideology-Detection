from __future__ import division
import cPickle
import pickle
from random import randint
import math
import naiveBayesOnlySentence
import logRegOnlySentence
import logRegw2v
import mostCommon
import logRegSentenceAndPhrase
import logRegOnlyPhrase
import naiveBayesSentenceAndPhrase
import run_w2vec
from sklearn.model_selection import KFold
from numpy import array, mean, median
from random import shuffle
import getStats

TRAIN_TEST_SPLIT = 0.9

def test_random(testData):
    count = 0
    for each_ob in testData:
        gen = randint(0, 1)
        if(gen == each_ob["label"]):
            count += 1
    print "Random Classifier:", count / len(testData)

# def train_test_split(lib, con):
#     total_data = []
#     for lib_sent in lib:
#        total_data.append({"sentence": lib_sent.get_words(), "label": 0})
#     for con_sent in con:
#       total_data.append({"sentence": con_sent.get_words(), "label": 1})
#     lim = int(math.floor(TRAIN_TEST_SPLIT * len(total_data)))
#     shuffle(total_data)
#     train_data = total_data[:lim]
#     test_data = total_data[lim:]
#
#     return total_data, train_data, test_data

def lr2(model, total_data):
    lim = int(math.floor(TRAIN_TEST_SPLIT * len(total_data)))
    train = total_data[:lim]
    test = total_data[lim:]
    results = {"accuracy":[], "fpr":[], "tpr":[]}


    accuracy, fpr, tpr = model.run(train, test)
    results["accuracy"].append(accuracy)
    results["fpr"].append(fpr)
    results["tpr"].append(tpr)

    print "Average Accuracy: ", mean(results["accuracy"])
    model.generate_plot(accuracy, fpr, tpr)

def lrw2v(model, total_data):
    lim = int(math.floor(TRAIN_TEST_SPLIT * len(total_data)))
    train = total_data[:lim]
    test = total_data[lim:]

    print len(train)


    f_vecs_train = pass_data_to_w2vec(train)
    f_vecs_test = pass_sent_to_w2vec(test)

    results = {"accuracy":[], "fpr":[], "tpr":[]}

    accuracy, fpr, tpr = model.run(f_vecs_train, f_vecs_test)
    results["accuracy"].append(accuracy)
    results["fpr"].append(fpr)
    results["tpr"].append(tpr)

    print "Average Accuracy: ", mean(results["accuracy"])
    model.generate_plot(accuracy, fpr, tpr)

def train_test_split(lib, con):
    total_data = []
    labels = {"Liberal":0, "Conservative":1}
    for lib_sent in lib:
       phrase_data = []
       for node in lib_sent:
           if hasattr(node, 'label') and node.label != 'Neutral':
               phrase_data.append([node.get_words(),labels[node.label]])
       total_data.append({"sentence": lib_sent.get_words(), "label": 0, "phrases":phrase_data })
    for con_sent in con:
      for node in con_sent:
           if hasattr(node, 'label') and node.label != 'Neutral':
               phrase_data.append([node.get_words(),labels[node.label]])
      total_data.append({"sentence": con_sent.get_words(), "label": 1, "phrases":phrase_data })

    lim = int(math.floor(TRAIN_TEST_SPLIT * len(total_data)))
    shuffle(total_data)
    train_data = total_data[:lim]
    test_data = total_data[lim:]
    return total_data, train_data, test_data

def k_fold_cross_validation(model, train_data, n_folds=10):
    kf = KFold(n_folds, shuffle=True)
    results = {"accuracy":[], "fpr":[], "tpr":[]}
    train_data = array(train_data)
    for traincv, testcv in kf.split(train_data):
        accuracy, fpr, tpr = model.run(train_data[traincv], train_data[testcv])
        results["accuracy"].append(accuracy)
        results["fpr"].append(fpr)
        results["tpr"].append(tpr)

    print "Average Accuracy: ", mean(results["accuracy"])
    model.generate_plot(accuracy, fpr, tpr)

def get_median(lst):
    return median(array(lst))

def get_corpus(lib, con):
    corpus = []
    lib_result = []
    con_result = []
    for each_lib in lib:
        corpus.append(each_lib.get_words())
        lib_result.append(each_lib.get_words())
    for each_con in con:
        corpus.append(each_con.get_words())
        con_result.append(each_con.get_words())

    return corpus, lib_result, con_result

def get_statistics(total_corpus, lib_result, con_result):
    getStats.run(total_corpus, lib_result, con_result)
    getStats.top_n(40)

def generate_files_for_cnn(data, flag):
    f1 = open(flag+"_lib", "w")
    f2 = open(flag+"_con", "w")

    count = 0

    for each in data:
        count += 1
        if(each["label"] == 0):
            f1.write(each["sentence"])
            f1.write("\n")
        if(each["label"] == 1):
            f2.write(each["sentence"])
            f2.write("\n")

    print count

def pass_sent_to_w2vec(data):
    feature_vecs = []
    for each in data:
        featureVec = run_w2vec.makeFeatureVec(each["sentence"].split(), 300 )
        feature_vecs.append({"sentence": featureVec, "label": each["label"]})
    return feature_vecs

def pass_data_to_w2vec(data):
    print "in pass_data_to_w2vec"
    count = 0
    feature_vecs = []
    for each in data:
        for phrase in each["phrases"]:
            count += 1
            featureVec = run_w2vec.makeFeatureVec(phrase[0].split(), 300 )
            feature_vecs.append({"sentence": featureVec, "label": phrase[1]})
            print count
    #print feature_vecs
    return feature_vecs

def pickle_f_vecs(f_vecs):
    print "in_pickle"
    fp = open("f_vecs.pkl", "wb")
    pickle.dump(f_vecs, fp)
    fp.close()

def load_pickle(f_name = "f_vecs.pkl"):
    f_vecs = pickle.load(open(f_name, "rb"))
    return f_vecs


if __name__ == '__main__':
    [lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))

    total_corpus, lib_result, con_result = get_corpus(lib, con)

    get_statistics(total_corpus, lib_result, con_result)
    total_data, train_data, test_data = train_test_split(lib, con)

    f_vecs = pass_data_to_w2vec(total_data)
    pickle_f_vecs(f_vecs)

    total_data = array(total_data)
    train_data = array(train_data)
    test_data = array(test_data)

    f_vecs = pass_data_to_w2vec(total_data)
    pickle_f_vecs(f_vecs)

    f_vecs = array(load_pickle())

    k_fold_cross_validation(logRegw2v, f_vecs, n_folds=10)


    test_random(test_data)
    k_fold_cross_validation(naiveBayesSentenceAndPhrase, total_data, n_folds=10)
    k_fold_cross_validation(logRegOnlySentence, total_data, n_folds=10)
    k_fold_cross_validation(mostCommon, total_data, n_folds=10)
    k_fold_cross_validation(logRegSentenceAndPhrase, total_data, n_folds=10)
    k_fold_cross_validation(logRegOnlyPhrase, total_data, n_folds=10)

    lr2(logRegOnlyPhrase, total_data)
    lrw2v(logRegw2v, total_data)
