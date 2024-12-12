#from textattack.shared import WordEmbedding
from .word_embedding import WordEmbedding
import numpy as np
import pandas as pd
def get_embedding(npz_file, words):
    arr = np.load(npz_file)["vecs"]
    word_map = {}
    idx_map = {}
    for i, word in enumerate(words):
        word_map[word] = i
        idx_map[i] = word
    return WordEmbedding(arr, word_map, idx_map)

from sklearn import neighbors as snb

def NNCalculator(data_matrix, metric, n_neighbors=1, algorithm='brute', drop_self=True, return_distance=False):
    """ Calculate the indices of up to N nearest neighbors.
    """
    
    if return_distance:
      n_neighbors += 1
    nn_calc = snb.NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm, n_jobs=-1)
    knn_fit = nn_calc.fit(data_matrix)
    knn_idx = knn_fit.kneighbors(data_matrix, n_neighbors=n_neighbors, return_distance=return_distance)
    if drop_self:
        knn_idx = knn_idx[:, 1:]
        
    return knn_idx


def CODER_embedding(n_neighbors, ent_type="diseases"):

  if ent_type == "drugs":

    entities = pd.read_csv("../qstab/external/FDA_Approved.csv", names=['drug_index','name'])
    embedding_matrix = np.load("../qstab/external/drug_word_vecs_umlsBERT.npz")['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding
  
  elif ent_type == "diseases":
    
    entities = pd.read_csv("../qstab/external/CTD_unique_disease_names.csv")
    embedding_matrix = np.load("../qstab/external/disease_word_vecs_umlsBERT.npz")['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding

def GTE_embedding(n_neighbors, ent_type="diseases"):

  if ent_type == "drugs":

    entities = pd.read_csv("../qstab/external/FDA_Approved.csv", names=['drug_index','name'])
    embedding_matrix = np.load("../qstab/external/drug_word_vecs_gtebase.npz")['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding
  
  elif ent_type == "diseases":
    
    entities = pd.read_csv("../qstab/external/CTD_unique_disease_names_cleaned.csv")
    embedding_matrix = np.load("../qstab/external/disease_word_cleaned_vecs_gtebase.npz")['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding