from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import nltk
import csv
import re
import pickle
stop_words = set(nltk.corpus.stopwords.words('english'))


def tokenize_text(tweet):
    tweet = re.sub(r'http\S+|https\S+', '', tweet)
    tokens = []
    for word in tweet.split(" "):
        if len(word) < 3 or word in stop_words:
            continue
        tokens.append(word)
    return tokens


def train_proccess(file_path):
    words = set([])
    count_iPhone = nltk.defaultdict(lambda: 0)
    count_Android = nltk.defaultdict(lambda: 0)
    word_Final = nltk.defaultdict(lambda: 0)
    time_iPhone = nltk.defaultdict(lambda: 0)
    time_Android = nltk.defaultdict(lambda: 0)
    time_final = {}

    # ToList
    real_twits = list(csv.reader(open(file_path), delimiter="\n"))
    for i in range(len(real_twits)):
        real_twits[i] = real_twits[i][0].split("\t")

    # Save good tweets
    for row in real_twits:
        if row[4] != "iphone" and row[4] != "android":
            if row[2].startswith('"@'):
                row[4] = 'android'
            elif "http" in row[2]:
                row[4] = 'iphone'
            elif re.findall(r'(\s\d{1,2}\s?(?:am|pm))', row[2].lower()):
                row[4] = 'iphone'

    # Only iPhone or Android
    real_twits = [row[2:] + [tokenize_text(row[2])] for row in real_twits if
                  row[1] == "realDonaldTrump" and (row[4] == "iphone" or row[4] == "android")]

    # Count words
    for row in real_twits:
        if row[0].startswith('"@'):
            row[3] = []
            continue
        for word in row[3]:
            words.add(word)
            if row[2] == 'android':
                count_Android[word] += 1
            else:
                count_iPhone[word] += 1

    sum_Android = sum(count_Android.values())
    sum_iPhone = sum(count_iPhone.values())

    for word in words:
        word_Final[word] = np.log2(((count_Android[word] + 1) / sum_Android) / ((count_iPhone[word] + 1) / sum_iPhone))

    count_Final = dict(word_Final)

    with open('words_price', 'wb') as f:
        pickle.dump(count_Final, f)

    # Count time
    for row in real_twits:
        time_tweet = datetime.strptime(row[1].split(" ")[1], '%H:%M:%S')
        if row[2] == 'android':
            time_Android[(time_tweet + timedelta(hours=time_tweet.minute // 30)).time().hour] += 1
        else:
            time_iPhone[(time_tweet + timedelta(hours=time_tweet.minute // 30)).time().hour] += 1

    sumtime_Android = sum(time_Android.values())
    sumtime_iPhone = sum(time_iPhone.values())

    for i in range(24):
        time_final[i] = np.log2((((1 + time_Android[i]) / sumtime_Android) / ((1 + time_iPhone[i]) / sumtime_iPhone)))

    with open('times_price', 'wb') as f:
        pickle.dump(time_final, f)

    # Build DataFrame
    list_feat = []
    for row in real_twits:
        features = {}

        # Quoted
        if row[0].startswith('"'):
            features["retweet"] = 1
            if row[0].count('"') % 2 == 1:
                row[0] = ''
        else:
            features["retweet"] = 0

        # am/pm
        if re.findall(r'(\s\d{1,2}\s?(?:am|pm))', row[0].lower()):
            features["am/pm"] = 1
        else:
            features["am/pm"] = 0

        # Time
        time_tweet = datetime.strptime(row[1].split(" ")[1], '%H:%M:%S')
        features["time"] = time_final[(time_tweet + timedelta(hours=time_tweet.minute // 30)).time().hour]

        # Links
        if "http" in row[0]:
            features["link"] = 1
        else:
            features["link"] = 0

        if row[2] == 'android':
            features["phone"] = 0
        else:
            features["phone"] = 1

        # Count words
        features["Count"] = sum([word_Final[word] for word in row[3]])

        list_feat.append(features)

    df = pd.DataFrame(list_feat)

    X = df.drop(["phone"], axis=1)
    Y = df['phone']

    return X, Y


def test_proccess(file_path):

    word_Final = nltk.defaultdict(lambda: 0)

    with open('words_price', 'rb') as f:
        words_Final = pickle.load(f)

    for key in words_Final.keys():
        word_Final[key] = words_Final[key]

    with open('times_price', 'rb') as f:
        time_Final = pickle.load(f)

    # ToList
    real_twits = list(csv.reader(open(file_path), delimiter="\n"))
    for i in range(len(real_twits)):
        real_twits[i] = real_twits[i][0].split("\t")

    # Only iPhone or Android
    real_twits = [row[1:] + [tokenize_text(row[1])] for row in real_twits]

    # Build DataFrame
    list_feat = []
    for row in real_twits:
        features = {}

        # Quoted
        if row[0].startswith('"@'):
            features["retweet"] = 1
            row[0] = ''
            row[2] = []
        else:
            features["retweet"] = 0

        # am/pm
        if re.findall(r'(\s\d{1,2}\s?(?:am|pm))', row[0].lower()):
            features["am/pm"] = 1
        else:
            features["am/pm"] = 0

        # Time
        time_tweet = datetime.strptime(row[1].split(" ")[1], '%H:%M:%S')
        features["time"] = time_Final[(time_tweet + timedelta(hours=time_tweet.minute // 30)).time().hour]

        # Links
        if "http" in row[0]:
            features["link"] = 1
        else:
            features["link"] = 0

        # Count words
        features["Count"] = sum([word_Final[word] for word in row[2]])

        list_feat.append(features)

    df = pd.DataFrame(list_feat)

    return df


def train_best_model(file_path):
    X, Y = train_proccess(file_path)
    logreg = LogisticRegression(C=10, penalty='l1', solver='liblinear')
    logreg.fit(X, Y)
    return logreg


def load_best_model():
    with open('LG_pickle', 'rb') as f:
        LG = pickle.load(f)
    return LG


def predict(m, fn):
    df = test_proccess(fn)
    return m.predict(df)





