import nltk
import random
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

posts = nltk.corpus.nps_chat.xml_posts([
    '10-19-20s_706posts.xml',
    '11-08-20s_705posts.xml',
    '11-09-20s_706posts.xml'
])


def train_classifier():
    pipeline = Pipeline([
        ('vect', HashingVectorizer()),  # CountVectorizer
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),  # LinearSVC
    ])
    
    # SVM with a Linear Kernel and default parameters
    # svm.SVC(kernel='linear')

    featuresets = [(post.text.replace('?', ' QQ'), 'Question' in post.get('class'))
                   for post in posts]
    random.shuffle(featuresets)
    size = int(len(featuresets) * .1)
    train_set, test_set = featuresets[size:], featuresets[:size]

    X, y = zip(*train_set)
    pipeline.fit(X, y)

    X, y = zip(*test_set)
    pred = pipeline.predict(X)

    pprint([z for z in zip(X, pred, y)
            if z[1] != z[2]])

    print 'accuracy %f' % pipeline.score(X, y)
    print classification_report(y, pred)

    return pipeline


classifier = train_classifier()
