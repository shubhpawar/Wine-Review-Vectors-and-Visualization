import nltk
import re
import sklearn.manifold
import multiprocessing
import pandas as pd
import seaborn as sns
import gensim.models.word2vec as w2v
import nltk.tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn.decomposition as dcmp

data = pd.read_excel('USWineData.xlsx',header=0)

data['description1'] = data['description'].map(str) +' '+ data['winery'].map(str) + '\t'

wineries = data.winery.unique().tolist()

descriptions = data['description1']

tokens = []
stop_words = set(stopwords.words('english'))
line1 = ''
corpus_raw = ''
for i in descriptions.tolist():
    line = word_tokenize(str(i))
    tokens = ' '.join(e for e in line if e.isalnum() and not e in stop_words)
    tokens = tokens + ' . '
    corpus_raw += tokens

print(corpus_raw[:1000])

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

print(raw_sentences[:5])

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(sentences[:5])

sentences_tagged = []
adjectives = []
for i in range(0,len(sentences)):
    s = []
    tagged=nltk.pos_tag(sentences[i])
    for i in tagged:
        if 'NN' in i[1] or 'JJ' in i[1] or i[0] in wineries:
            s.append(i[0])
        if 'JJ' == i[1]:
            adjectives.append(i[0])
    sentences_tagged.append(s)
    

sentences = sentences_tagged

print(sentences[:10])

num_features = 300
min_word_count = 1
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-3
seed=1993

wine2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

wine2vec.build_vocab(sentences)

wine2vec.train(sentences, total_examples=wine2vec.corpus_count,epochs=wine2vec.iter)

all_word_vectors_matrix = wine2vec.wv.syn0

pca = dcmp.PCA(n_components=10)
data = pca.fit_transform(all_word_vectors_matrix)

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix_2d = tsne.fit_transform(data)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[wine2vec.wv.vocab[word].index])
            for word in wine2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)
sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

plot_region(x_bounds=(-10, 0), y_bounds=(-10, 0))

wineries = [e for e in wineries if str(e).isalnum()]

print(wineries)

d = {'winery':[],'adjective':[],'similarity':[]}

for wine in wineries[:20]:
    for adjective in adjectives[:500]:
        if wine in wine2vec.wv.vocab.keys() and adjective in wine2vec.wv.vocab.keys() and wine2vec.similarity(wine, adjective)>=0.5:
            
            d['winery'].append(wine)
            d['adjective'].append(adjective)
            d['similarity'].append(wine2vec.similarity(wine, adjective))

df = pd.DataFrame(data=d)

writer = pd.ExcelWriter('data.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()
