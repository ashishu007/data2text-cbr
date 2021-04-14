import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

# Let's define some utility functions
def similarity_between_docs(doc1, doc2):
    v1 = np.reshape(doc1, (1, -1))
    v2 = np.reshape(doc2, (1, -1))
    return cosine_similarity(v1, v2)[0][0]

def return_cent(cents, emb):
    sims = {idx:similarity_between_docs(emb, i) for idx, i in enumerate(cents)}
    mx, mx_id = 0, 0
    for k, v in sims.items():
        if v > mx:
            mx = v
            mx_id = k
    return mx_id


ordinal_set = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
                    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
                    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10, 
                    "11th": 11, "12th": 12, "13th": 13, "14th": 14, "15th": 15, "last": 15, "atop":1}

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
centers = np.load('./clustering/data/train_pos_clust_centers.npy')
center_names = np.load('./clustering/data/train_pos_clust_names.npy')

data = pd.read_csv('./clustering/data/abstract_sentences.csv')
print(data.shape)
abs_sents = data['abs']

print('encoding bitch')
sentence_embeddings = model.encode(abs_sents)
x_norm = preprocessing.normalize(sentence_embeddings)
print('encoding done bitch')

sents_w_clusts = {
    'sent': [],
    'abs': [],
    'clust': [],
    'game_idx': [],
    'season': [],
    'sent_wo_ord': []
}

for idx, i in enumerate(x_norm):
    name_id = return_cent(centers, i)
    name = center_names[name_id]
    orig_sent = data['coref_sent'][idx]

    sents_w_clusts['clust'].append(name)
    sents_w_clusts['sent'].append(orig_sent)
    sents_w_clusts['abs'].append(data['abs'][idx])
    sents_w_clusts['game_idx'].append(data['game_idx'][idx])
    sents_w_clusts['season'].append(data['season'][idx])

    new_orig_sent = ''
    for tok in orig_sent.split(' '):
        if tok in ordinal_set:
            new_orig_sent += f'{ordinal_set[tok]} '
        else:
            new_orig_sent += f'{tok} '

    sents_w_clusts['sent_wo_ord'].append(new_orig_sent)

df_final = pd.DataFrame(sents_w_clusts)
print(df_final.shape)
df_final.to_csv('./clustering/data/all_clusters.csv', index=0)
