from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy import array

def run(f_vecs_train, f_vecs_test):
    # Generate counts from text using a vectorizer.
    # This performs our step of computing word counts.
    print "this is run"
    train_features = [f["sentence"] for f in f_vecs_train]
    test_features = [t["sentence"] for t in f_vecs_test]
    actual = [t["label"] for t in f_vecs_test]
    labels = [f["label"] for f in f_vecs_train]
    # Fit a model to the training data.
    lr = LogisticRegression()
    lr.fit(train_features, labels)
    print "done"
    predictions = lr.predict(test_features)

    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions)
    print("Logistic Regression AUC: {0}".format(metrics.auc(fpr, tpr)))
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, fpr, tpr

def generate_plot(roc_auc, fpr, tpr):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
