from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.shared import AttackedText

# class AnswerConstraint(PreTransformationConstraint):
#     def __init__(self, df, json_entities, tui):
#         self.df = df
#         self.entity = json_entities
#         self.tui = tui
#         self.ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}

#     def _get_modifiable_indices(self, current_text):
#         idx_set = set()
#         words = current_text.words
#         # if "?" not in current_text.text:
#         #    return set()
#         question, choices = current_text.text.split("\nA:")
#         question = AttackedText(question)
#         choices = choices.split("\n")
#         choices = [i for i in choices if len(i) != 0]
#         offsets = [len(question.words) + 1]
#         for i in range(len(choices)):
#             choices[i] = AttackedText(choices[i])
#         for i in range(1, len(choices)):
#             offsets.append(offsets[-1] + len(choices[i-1].words))
#         row = self.df[self.df["formatted_input"] == current_text.text]
#         row_idx = row["row_num"].astype(int).item()
#         ans_idx = row["label"].astype(int).item()
#         ans = row["answer"]
#         json_entities = self.entity[row_idx]
#         for d in json_entities:
#             index = self.ans_map[d]            
#             if index != ans_idx:
#                 valid_entities = []
#                 entities = json_entities[d]
#                 tui = entities["labels"]
#                 entity = entities["text"]
#                 for e, t in zip(entity, tui):
#                     if t in self.tui:
#                         valid_entities.append(e)
#                 for e in valid_entities:
#                     idx_list = self.phrase_to_indices(e, choices[index].words, offsets[index])
#                     idx_set.update(idx_list)
#         return idx_set
    
#     def phrase_to_indices(self, phrase, word_list, offset):
#         phrase_words = phrase.split()
#         phrase_length = len(phrase_words)
#         indices = []

#         for i in range(len(word_list) - phrase_length + 1):
#             if word_list[i:i + phrase_length] == phrase_words:
#                 indices.append(i + offset)

#         return indices


class AnswerConstraint(PreTransformationConstraint):
    def __init__(self, df, json_entities, tui):
        self.df = df
        self.entity = json_entities
        self.tui = tui
        self.ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def _get_modifiable_indices(self, current_text):
        idx_set = set()
        words = current_text.words
        # if "?" not in current_text.text:
        #    return set()
        question, choices = current_text.text.split("\nA:")
        question = AttackedText(question)
        choices = choices.split("\n")
        choices = [i for i in choices if len(i) != 0]
        offsets = [len(question.words) + 1]
        for i in range(len(choices)):
            choices[i] = AttackedText(choices[i])
        for i in range(1, len(choices)):
            offsets.append(offsets[-1] + len(choices[i-1].words))
        row = self.df[self.df["formatted_input"] == current_text.text]
        row_idx = row["row_num"].astype(int).item()
        ans_idx = row["label"].astype(int).item()
        ans = row["answer"]
        json_entities = self.entity[row_idx]
        valid_entities = []
        valid_chids = []
        valid_boundaries = []
        
        for d in json_entities:
            index = self.ans_map[d]            
            
            if index != ans_idx:
                entities = json_entities[d]
                tui = entities["labels"]
                entity = entities["text"]
                boundary = entities["boundaries"]
                # print(tui, entity, boundary)
                
                for e, t, b in zip(entity, tui, boundary):
                    if t in self.tui:
                        # print("selected include {}".format(e, t, b))
                        valid_entities.append(e)
                        valid_chids.append(d)
                        valid_boundaries.append(b)
                        # print(valid_entities, valid_chids, valid_boundaries)
                # for e in valid_entities:
                #     idx_list = self.phrase_to_indices(e, choices[index].words, offsets[index])
                #     idx_set.update(idx_list)
        # print(valid_entities, valid_chids, valid_boundaries)
        return valid_entities, valid_chids, valid_boundaries