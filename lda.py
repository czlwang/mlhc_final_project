import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import LatentDirichletAllocation
import codecs
from mappings import icd_map
from sklearn.manifold import TSNE

lookup_dir = '../data/lookup'
if not lookup_dir in sys.path:
    sys.path.append(lookup_dir)

cpt_df = pd.read_csv("/home/project/data/cpt_full.csv")
cpt_df.columns = ["code", "label"]

doc = codecs.open("/home/project/data/ICD9/output/output.csv",'rU','latin-1')
icd9_table = pd.read_csv(doc)
icd9_table.head()

def fit_lda(X, n):
	'''
	Fit an LDA model with n topics to data X 
	'''
	lda = LatentDirichletAllocation(n_components=n, random_state=0)
	lda.fit(X)
	return lda

def cpt_lookup(cpt):
	'''
	Return string description for cpt code
	'''
    if len(cpt_df[cpt_df['code'] == str(cpt)]["label"].unique())>0:
        return cpt_df[cpt_df['code'] == str(cpt)]["label"].unique()[0]
    return cpt

def icd_lookup(icd):
	'''
	Return string description for ICD-9 code
	'''
    icd = str(icd)
    icd = icd.replace('.', '')
    formatted = '%s.%s' % (icd[:3],icd[3:])
    if formatted in icd_map:
        return icd_map[formatted]
    else:
        return icd

def display_topics(model, feature_names, no_top_words):
	'''
	Print the descriptions of the top words with 
	the most probability weight for the topics in the LDA model
	'''
    topic_descriptions = []
    for topic_idx, topic in enumerate(model.components_):
        print ("\n----Topic %d:-----" % (topic_idx))
        words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_description = []
        for word in words:
            if word[1] == 'icd':
                print("icd", icd_lookup(word[2]))
                topic_description.append(("icd", icd_lookup(word[2])))
            elif word[1] == "ndc":
                print("ndc", ndc_lookup(word[2]))
                topic_description.append(("ndc", ndc_lookup(word[2])))
            elif word[1] == 'cpt':
                print("cpt", word[2], cpt_lookup(word[2]))
                topic_description.append(("cpt", word[2], cpt_lookup(word[2])))
            elif word[0] == 'demographics':
                print(word[1], word[2])
                topic_description.append((word[1], word[2]))
            else:
                print(word)
        topic_descriptions.append(topic_description)
    return topic_descriptions

def plot_topics(model, feature_names, no_top_words):
	'''
	Plot probability of the top words with the 
	most probability weight for the topics in the LDA model
	'''
    for topic_idx, topic in enumerate(model.components_):
        print ("\n----Topic %d:-----" % (topic_idx))
        words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        weights = sorted(topic)[:-no_top_words - 1:-1]
        descriptions = []
        
        for word in words:
            if word[1] == 'icd':
                descriptions.append(icd_lookup(word[2]))
            elif word[1] == "ndc":
                descriptions.append(ndc_lookup(word[2])[0])
            elif word[1] == 'cpt':
                descriptions.append(cpt_lookup(word[2]))
            elif word[0] == 'demographics':
                descriptions.append(" ".join(word[1:]))
            else:
                print(word)
            
        word_weights = zip(descriptions, weights)

        fig, ax = plt.subplots()
        index = np.arange(no_top_words)
        bar_width = 0.35
        rects1 = ax.bar(index, weights, bar_width)
        ax.set_ylabel('Weights')
        ax.set_xticks(index)
        ax.set_xticklabels(descriptions)
        plt.xticks(rotation=90)
        plt.show()

def plot_enrichment(model, feature_names, no_top_words, X, y):
	'''
	Plot the enrichment of the topics from an LDA model and a given outcome vector y
	'''
    transformed = model.transform(X)
    denom = np.sum(transformed, axis=0)
    num = np.sum(transformed*y[:, np.newaxis], axis=0)
    enrichment = num/denom
    baseline = sum(y)/len(y)
    
    fig, ax = plt.subplots()
    index = np.arange(n)

    markerline, stemlines, baseline = ax.stem(enrichment, bottom=baseline)
    ax.set_ylabel('Enrichment')
    ax.set_xticks(index)
    ax.set_xticklabels(list(["topic " + str(i) for i in range(n)]))
    plt.xticks(rotation=90)
    plt.setp(baseline, color='r', linewidth=2)
    plt.show()

    print(enrichment)

def plot_tsne(lda, X):
	lda.fit(X)
	lda_x = lda.transform(X)
	cohorts_train = np.argmax(lda_x, axis=1).astype(str)
	X_embedded = TSNE(n_components=2).fit_transform(lda_x)
	x = X_embedded[:,0]
	y = X_embedded[:,1]
	colors = cohorts_train.astype(int)
	plt.scatter(x, y, c=colors, alpha=0.5)
