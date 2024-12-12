from collections import defaultdict
import csv
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import neighbors as snb
import torch

from textattack.shared import utils
from textattack.shared import AbstractWordEmbedding

class WordEmbedding(AbstractWordEmbedding):
  """Object for loading word embeddings and related distances for TextAttack.
    This class has a lot of internal components (e.g. get consine similarity)
    implemented. Consider using this class if you can provide the appropriate
    input data to create the object.

    Args:
        emedding_matrix (ndarray): 2-D array of shape N x D where N represents size of vocab and D is the dimension of embedding vectors.
        word2index (Union[dict|object]): dictionary (or a similar object) that maps word to its index with in the embedding matrix.
        index2word (Union[dict|object]): dictionary (or a similar object) that maps index to its word.
        nn_matrix (ndarray): Matrix for precomputed nearest neighbours. It should be a 2-D integer array of shape N x K
            where N represents size of vocab and K is the top-K nearest neighbours. If this is set to `None`, we have to compute nearest neighbours
            on the fly for `nearest_neighbours` method, which is costly.
    """

  PATH = "word_embeddings"

  def __init__(self, embedding_matrix, word2index, index2word, nn_matrix=None):
    self.embedding_matrix = embedding_matrix
    self._eps = np.finfo(self.embedding_matrix.dtype).eps
    self.normalized_embeddings = self.embedding_matrix / np.expand_dims(
        np.maximum(np.linalg.norm(embedding_matrix, ord=2, axis=-1), self._eps),
        1)
    self._word2index = word2index
    self._index2word = index2word
    self.nn_matrix = nn_matrix

    # Dictionary for caching results
    self._mse_dist_mat = defaultdict(dict)
    self._cos_sim_mat = defaultdict(dict)
    self._nn_cache = {}

  def __getitem__(self, index):
    """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
    if isinstance(index, str):
      try:
        index = self._word2index[index]
      except KeyError:
        return None
    try:
      return self.embedding_matrix[index]
    except IndexError:
      # word embedding ID out of bounds
      return None

  def word2index(self, word):
    """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
    return self._word2index[word]

  def index2word(self, index):
    """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)

        """
    return self._index2word[index]

  def get_mse_dist(self, a, b):
    """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
    if isinstance(a, str):
      a = self._word2index[a]
    if isinstance(b, str):
      b = self._word2index[b]
    a, b = min(a, b), max(a, b)
    try:
      mse_dist = self._mse_dist_mat[a][b]
    except KeyError:
      e1 = self.embedding_matrix[a]
      e2 = self.embedding_matrix[b]
      e1 = torch.tensor(e1).to(utils.device)
      e2 = torch.tensor(e2).to(utils.device)
      mse_dist = torch.sum((e1 - e2)**2).item()
      self._mse_dist_mat[a][b] = mse_dist

    return mse_dist

  def get_cos_nn(self, query_point: np.ndarray, topn: int):
    """Finds the nearest neighbors to the query point using cosine similarity.

    Args:
      query_point: The point in space of which we want to find nearest neighbors
        <float32/64>[1, embedding_size]
      topn: This controls how many neighbors to return
    Returns:
      A list of tokens in the embedding space.
      A list of distances.
    """

    normalizer = max(np.linalg.norm(query_point, ord=2),
                     np.finfo(query_point.dtype).eps)
    query_point = query_point / normalizer

    cosine_similarities = np.matmul(query_point, self.normalized_embeddings.T)
    if topn == 1:
      nearest_neighbors = list([np.argsort(cosine_similarities)[-1]])
    else:
      # argsort sorts lowest to highest, we want the largest values
      nearest_neighbors = list(np.argsort(cosine_similarities)[-topn:])
    nearest_neighbors.reverse()
    distance_list = list(cosine_similarities[nearest_neighbors])
    nearest_tokens = [self.index2word(index) for index in nearest_neighbors]
    return nearest_tokens, distance_list

  def get_euc_nn(self, query_point: np.ndarray, topn: int):
    """Finds the nearest neighbors to the query point using cosine similarity.

    Args:
      query_point: The point in space of which we want to find nearest neighbors
        <float32/64>[1, embedding_size]
      topn: This controls how many neighbors to return
    Returns:
      A list of tokens in the embedding space.
      A list of distances.
    """
    euclidean_distances = np.linalg.norm(self.embedding_matrix - query_point,
                                         axis=-1,
                                         ord=2)

    if topn == 1:
      nearest_neighbors = list([np.argsort(euclidean_distances)[0]])
    else:
      # argsort sorts lowest to highest, we want the smallest distance
      nearest_neighbors = list(np.argsort(euclidean_distances)[:topn])
    nearest_tokens = [self.index2word(index) for index in nearest_neighbors]
    distance_list = list(euclidean_distances[nearest_neighbors])
    return nearest_tokens, distance_list

  def get_cos_sim(self, a, b):
    """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
    if isinstance(a, str):
      a = self._word2index[a.lower()]
    if isinstance(b, str):
      b = self._word2index[b.lower()]
    a, b = min(a, b), max(a, b)
    try:
      cos_sim = self._cos_sim_mat[a][b]
    except KeyError:
      e1 = self.embedding_matrix[a]
      e2 = self.embedding_matrix[b]
      e1 = torch.tensor(e1).to(utils.device)
      e2 = torch.tensor(e2).to(utils.device)
      cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).item()
      self._cos_sim_mat[a][b] = cos_sim
    return cos_sim

  def nearest_neighbours(self, index, topn):
    """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
    if isinstance(index, str):
      index = self._word2index[index]
    if self.nn_matrix is not None:
      nn = self.nn_matrix[index][1:(topn + 1)]
    else:
      try:
        nn = self._nn_cache[index]
      except KeyError:
        embedding = torch.tensor(self.embedding_matrix).to(utils.device)
        vector = torch.tensor(self.embedding_matrix[index]).to(utils.device)
        dist = torch.norm(embedding - vector, dim=1, p=None)
        # Since closest neighbour will be the same word, we consider N+1 nearest neighbours
        nn = dist.topk(topn + 1, largest=False)[1][1:].tolist()
        self._nn_cache[index] = nn

    return nn

  @staticmethod
  def embeddings_from_file(path_to_embeddings):
    """Given a csv file using spaces as delimiters, use the first column
    as the vocabulary and the rest of the columns as the word embeddings."""
    embedding_file = open(path_to_embeddings)
    lines = embedding_file.readlines()
    vocab = []
    vectors = []
    for line in lines:
      if line == "":
        break
      line = line.split()
      vocab.append(line[0])
      vectors.append([float(value) for value in line[1:]])
    embedding_matrix = np.array(vectors)
    word2index = {}
    index2word = {}
    for i, token in enumerate(vocab):
      word2index[token] = i
      index2word[i] = token

    embedding = WordEmbedding(embedding_matrix, word2index, index2word)
    
    return embedding

def counterfitted_GLOVE_embedding():
    """Returns a prebuilt counter-fitted GLOVE word embedding proposed by
    "Counter-fitting Word Vectors to Linguistic Constraints" (Mrkšić et
    al., 2016)"""
    if (
        "textattack_counterfitted_GLOVE_embedding" in utils.GLOBAL_OBJECTS
        and isinstance(
            utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"],
            WordEmbedding,
        )
    ):
        # avoid recreating same embedding (same memory) and instead share across different components
        return utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"]

    word_embeddings_folder = "paragramcf"
    word_embeddings_file = "paragram.npy"
    word_list_file = "wordlist.pickle"
    mse_dist_file = "mse_dist.p"
    cos_sim_file = "cos_sim.p"
    nn_matrix_file = "nn.npy"

    # Download embeddings if they're not cached.
    word_embeddings_folder = os.path.join(
        WordEmbedding.PATH, word_embeddings_folder
    ).replace("\\", "/")
    word_embeddings_folder = utils.download_from_s3(word_embeddings_folder)
    # Concatenate folder names to create full path to files.
    word_embeddings_file = os.path.join(
        word_embeddings_folder, word_embeddings_file
    )
    word_list_file = os.path.join(word_embeddings_folder, word_list_file)
    mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
    cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
    nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

    # loading the files
    embedding_matrix = np.load(word_embeddings_file)
    word2index = np.load(word_list_file, allow_pickle=True)
    index2word = {}
    for word, index in word2index.items():
        index2word[index] = word
    nn_matrix = np.load(nn_matrix_file)

    embedding = WordEmbedding(embedding_matrix, word2index, index2word, nn_matrix)

    with open(mse_dist_file, "rb") as f:
        mse_dist_mat = pickle.load(f)
    with open(cos_sim_file, "rb") as f:
        cos_sim_mat = pickle.load(f)

    embedding._mse_dist_mat = mse_dist_mat
    embedding._cos_sim_mat = cos_sim_mat

    utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"] = embedding

    return embedding


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

    entities = pd.read_csv("./external/FDA_Approved.csv", names=['drug_index','name'])
    embedding_matrix = np.load("./external/drug_word_vecs_umlsBERT.npz")['vecs']

    # drug_reps = np.load("../qstab/external/all_drug_word_vecs_umlsBERT.npz", allow_pickle=True)
    # dgnames = np.squeeze(drug_reps["names"])
    # entities = pd.DataFrame.from_dict({"name":dgnames.tolist()})
    # embedding_matrix = drug_reps['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding
  
  elif ent_type == "diseases":
    
    entities = pd.read_csv("./external/CTD_unique_disease_names_cleaned.csv")
    embedding_matrix = np.load("./external/disease_word_cleaned_vecs_umlsBERT.npz")['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding


def GTE_embedding(n_neighbors, ent_type="diseases"):

  if ent_type == "drugs":

    entities = pd.read_csv("./external/FDA_Approved.csv", names=['drug_index','name'])
    embedding_matrix = np.load("./external/drug_word_vecs_gtebase.npz")['vecs']
    # print(embedding_matrix.shape)
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding
  
  elif ent_type == "diseases":
    
    entities = pd.read_csv("./external/CTD_unique_disease_names_cleaned.csv")
    embedding_matrix = np.load("./external/disease_word_cleaned_vecs_gtebase.npz")['vecs']
    index2word = dict(zip(entities.index, entities['name']))
    word2index = dict(zip(entities['name'], entities.index))
    neighbor_indices = NNCalculator(embedding_matrix, n_neighbors=n_neighbors,
                                    metric='cosine', algorithm='brute')
    embedding = WordEmbedding(embedding_matrix, word2index, index2word, neighbor_indices)

    return embedding