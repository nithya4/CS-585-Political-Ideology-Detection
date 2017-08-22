from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

total_counts = defaultdict(float)
lib_words_counts = defaultdict(float)
con_words_counts = defaultdict(float)

def tokenize_doc(doc):
    '''
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    '''
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    lowered_tokens = [word for word in lowered_tokens if word not in stopwords.words('english')]
    for token in lowered_tokens:
        total_counts[token] += 1.0

def tokenize(doc):
    toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)

    tokens = toker.tokenize(doc)
    lowered_tokens = map(lambda t: t.lower(), tokens)
    lowered_tokens = [word for word in lowered_tokens if word not in stopwords.words('english')]
    return lowered_tokens


def top_n(n):
    print sorted(lib_words_counts.items(), key=lambda (w,c): -c)[:n]
    print sorted(con_words_counts.items(), key=lambda (w,c): -c)[:n]

def generate_lib_con_counts(lib_result, con_result):
    for each_sent in lib_result:
        tokens = tokenize(each_sent)
        for each in tokens:
            lib_words_counts[each] += 1
    for each_sent in con_result:
        tokens = tokenize(each_sent)
        for each in tokens:
            con_words_counts[each] += 1


def run(corpus, lib_result, con_result):
    # for each in corpus:
    #     tokenize_doc(each)
    generate_lib_con_counts(lib_result, con_result)
