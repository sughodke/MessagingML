import nltk
import csv
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

unique_tags = set()


def transform(post):
    tokens = nltk.word_tokenize(post['Title'])
    tagged_tokens = nltk.pos_tag(tokens)

    serialized = ['_'.join(z)
                  for z in tagged_tokens]

    text = ' '.join(serialized)
    # text = text.replace('?', ' QQ')

    tokenizer = nltk.tokenize.RegexpTokenizer('<[a-z-]\w+>')
    tags = tokenizer.tokenize(post['Tags'])
    unique_tags += set(tags)

    return text, tags


def train_classifier():
    with open('bicycle.csv', 'rb') as f:
        reader = csv.DictReader(f)
        featuresets = [transform(post) for post in reader]
    
    size = int(len(featuresets) * .1)
    train_set, test_set = featuresets[size:], featuresets[:size]

    per_tag = dict.fromkeys(
            unique_tags, 
            Pipeline([
                ('vect', HashingVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier())
            ])
        )
    
    for tag, pipeline in per_tag:
        X, y = zip(*train_set)
        y = [tag in v for v in y]
        pipeline.fit(X, y)

        X, y = zip(*test_set)
        y = [tag in v for v in y]
        pred = pipeline.predict(X)

        pprint([z for z in zip(X, pred, y)
                if z[1] != z[2]])

        print 'accuracy %f' % pipeline.score(X, y)
        print classification_report(y, pred)

    return pipeline


classifier = train_classifier()
