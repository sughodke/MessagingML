from __future__ import print_function

import nltk
import csv
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

unique_tags = set([])
cache = True

def transform(post):
    tokens = nltk.word_tokenize(post['Title'])
    tagged_tokens = nltk.pos_tag(tokens)

    serialized = ['_'.join(z)
                  for z in tagged_tokens]

    text = ' '.join(serialized)
    # text = text.replace('?', ' QQ')

    tokenizer = nltk.tokenize.RegexpTokenizer('<[a-z-]\w+>')
    tags = tokenizer.tokenize(post['Tags'])
    unique_tags.update(set(tags))

    return text, tags


def train_classifier():
    print('reading')
    with open('bicycle.csv', 'rb') as f:
        reader = csv.DictReader(f)
        featuresets = [transform(post) for post in reader]

    # featuresets = featuresets[:3000]
    size = int(len(featuresets) * .1)
    train_set, test_set = featuresets[size:], featuresets[:size]

    print('classifying')
    if cache:
        per_tag = joblib.load('models/tags.pkl')

    else:
        per_tag = dict.fromkeys(
            unique_tags,
            Pipeline([
                ('vect', HashingVectorizer(non_negative=True)),
                ('clf', SGDClassifier())
            ])
        )

        X, y = zip(*train_set)
        for tag, pipeline in per_tag.iteritems():
            y_tagged = [tag in v for v in y]
            try:
                pipeline.fit(X, y_tagged)
                print('.', end='')
            except ValueError:
                print('x', end='')
        joblib.dump(per_tag, 'models/tags.pkl', compress=True)

    print()
    print('verifying')
    output = {}
    for tag, pipeline in per_tag.iteritems():
        X, y = zip(*test_set)
        y = [tag in v for v in y]

        pred = pipeline.predict(X)

        pprint([z for z in zip(X, pred, y)
                if z[1] != z[2]])

        """
        for pred in pipeline.predict(X):
            if pred:
                output[' '.join(X)] = '%s %s' % (output.get(' '.join(X), ''), tag)
        """

        # print('%s accuracy %f' % (tag, pipeline.score(X, y)))

    pprint(output)

    return per_tag


classifier = train_classifier()
