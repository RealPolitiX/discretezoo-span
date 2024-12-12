from textattack.constraints import Constraint
import numpy as np
from numpy.linalg import norm
import pandas as pd
from difflib import Differ

class VectorScore(Constraint):
    def __init__(self, min_score, df, vec_file, compare_against_original=True):
        super().__init__(compare_against_original)
        self.min_score = min_score
        self.vecs = np.load(vec_file)["vecs"]
        self.words = df["name"].to_list()
        self.diff = Differ()
        
    def get_diff(self, starting_text, transformed_text):
        cand = transformed_text.text.split()
        ref = starting_text.text.split()
        cand = "\n".join(cand)
        ref = "\n".join(ref)
        diff = list(self.differ.compare(cand, ref))
    
        diff_phrases1 = []
        diff_phrases2 = []
        
        phrase1 = []
        phrase2 = []
        
        for word in diff:
            if word.startswith("- "):  # Word unique to str1
                phrase1.append(word[2:])
            elif word.startswith("+ "):  # Word unique to str2
                phrase2.append(word[2:])
            else:  # Word common to both strings
                if phrase1:
                    diff_phrases1.append(" ".join(phrase1))
                    phrase1 = []
                if phrase2:
                    diff_phrases2.append(" ".join(phrase2))
                    phrase2 = []
        
        # Capture any remaining phrases
        if phrase1:
            diff_phrases1.append(" ".join(phrase1))
        if phrase2:
            diff_phrases2.append(" ".join(phrase2))
        
        return diff_phrases1, diff_phrases2

    def similarity(self, diff1, diff2):
        coss = []
        for w1, w2 in zip(diff1, diff2):
            if w1 in self.words and w2 in self.words:
                i1 = self.words.index_of(w1)
                i2 = self.words.index_of(w2)
                v1 = self.vecs[i1]
                v2 = self.vecs[i2]
                cos = np.dot(v1, v2)/(norm(v1) * norm(v2))
                coss.append(cos)
        return np.array(coss)

    def _check_constraint(self, transformed_text, reference_text):
        
        diff1, diff2 = self.get_diff(reference_text, transformed_text)
        scores = self.similarity(diff1, diff2)
        return np.all(scores >= self.min_score)