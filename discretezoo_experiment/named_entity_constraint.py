from textattack.constraints import Constraint
import nltk
import functools
# #from pytrie import Trie
# nltk.download("punkt")  # The NLTK tokenizer
# nltk.download("maxent_ne_chunker")  # NLTK named-entity chunker
# nltk.download("words")  # NLTK list of words
# nltk.download("averaged_perceptron_tagger")

@functools.lru_cache(maxsize=2**14)
def get_entities(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    # Setting `binary=True` makes NLTK return all of the named
    # entities tagged as NNP instead of detailed tags like
    #'Organization', 'Geo-Political Entity', etc.
    entities = nltk.chunk.ne_chunk(tagged, binary=True)
    return entities.leaves()

class NamedEntityConstraint(Constraint):
    def __init__(self, compare_against_original, entity_set):
        super().__init__(True)
        self.entity_set = entity_set

    """A constraint that ensures `transformed_text` only substitutes named entities from `current_text` with other named entities."""
    def diff(self, a, b):
        return set(a) - set(b)
    
    @functools.lru_cache(maxsize=2**14)
    def trie_contains(self, word_list):
        while len(word_list) > 0:
            word = word_list.pop(0)
            if word in self.entity_set:
                if len(word_list) == 0:
                    return True
                new_first = word + " " + word_list[0]
                res = self.trie_contains(word_list)
                word_list[0] = new_first
                return res or self.trie_contains(word_list)
            elif len(word_list) == 0:
                return False
            else:
                new_first = word + " " + word_list[0]
                word_list[0] = new_first
                return self.trie_contains(word_list)
        return True
    
    def _check_constraint(self, transformed_text, current_text):
        t = transformed_text.text.split()
        c = current_text.text.split()
        
        #diff the two strings
        d1 = self.diff(t, c)
        d2 = self.diff(c, t)
        for word_list in [d1, d2]:
            if not self.trie_contains(word_list):
                return False
        return True