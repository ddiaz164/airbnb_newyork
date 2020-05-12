
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF as NMF_sklearn
def text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    stop_set = set(stopwords.words('english'))
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems
    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()
    return vectorizer, vocabulary
def softmax(v, temperature=1.0):
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s
def hand_label_topics(H, vocabulary):
    hand_labels = []
    for i, row in enumerate(H, start=1):
        top_five = np.argsort(row)[::-1][:5]
        #print('topic', i)
        #print('-->', ' '.join(vocabulary[top_five]))
        label = i
        hand_labels.append(label)
        #print()
    return hand_labels
def analyze(index, contents, W, hand_labels):
    probs = softmax(W[index], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        print('--> {:.2f}% {}'.format(prob * 100, label))
    print()
def analyze_probs(df, W, hand_labels):
    probs = []
    for i in df.index:
        prob = softmax(W[i], temperature=0.01)
        probs.append(prob)
    return probs
def analyze_all(df, W, hand_labels):
    labels = []
    for i in df.index:
        probs = softmax(W[i], temperature=0.01)
        labels.append(hand_labels[probs.argmax()])
    return labels
def scree_plot(ax, pca, n_components_to_plot=8, title=None):

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)
    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=12)
    #ax.set_xticklabels(ind, fontsize=12)
    #ax.set_ylim(0, max(vals) + 0.05)
    #ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)
def plot_name_embedding(ax, X, y, title=None):

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(digits.target[i]), 
                 color=plt.cm.Set1(y[i] / 10.), 
                 fontdict={'weight': 'bold', 'size': 12})
    ax.set_xticks([]), 
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])
    if title is not None:
        ax.set_title(title, fontsize=16)
        