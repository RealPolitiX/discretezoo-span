"""

Berger Discrete ZOO
=======================================
(Generating Natural Language Adversarial Examples)
"""

import textattack
from .constraints import STRICT_CONSTRAINTS, LAX_CONSTRAINTS, NONE_CONSTRAINTS
from textattack.constraints import PreTransformationConstraint
import pandas as pd
from .attack import Attack as Atk
from .untargeted_classification_loss import ZOOUntargetedClassification
from .simple_search_method import DiscreteZOO
from .word_swap_displacement import WordSwapEmbeddingDisplacement
from .word_embedding import WordEmbedding, counterfitted_GLOVE_embedding, CODER_embedding, GTE_embedding
from .named_entity_constraint import NamedEntityConstraint
from .answer_constraint import AnswerConstraint
from .encoding_vec import VectorScore
# from .custom_word_embedding import get_embedding, CODER_embedding
import json
import numpy as np

def Attack(model,
        named_entity = False,
        grad_steps = 1,
        constraint_mode = "strict",
        type = "drugs",
        candidates = 10,
        df_dataset = None,
        constraints = None,
        qa_dataset = "medmcqa",
        atk_embedding = "gte",
        cand_select = "random",
        expn = 0,
        threshold_samples = False,
        **kwargs):
    """Berger, N. Ebert, S. Sokolov, A. Riezler, S.
  """
    # candidate_number = kwargs.pop("candidates", 10)
    #embeddings_file = 'glove.6b.50d.filtered.txt'
    if constraint_mode == "strict":
        constraints = STRICT_CONSTRAINTS
    elif constraint_mode == "lax":
        constraints = LAX_CONSTRAINTS
    elif constraint_mode == "none":
        constraints = NONE_CONSTRAINTS
    if named_entity:
        fpre = r"/home/rpxian/Code/repos/discretezoo"
        if type == "drugs":
            df = pd.read_csv(fpre + r"/external/FDA_Approved.csv", names=['drug_index', 'name'])
            tui = ['T116', 'T195', 'T123', 'T122', 'T103', 'T120', 'T104',
                'T200', 'T196', 'T126', 'T131', 'T125', 'T129', 'T130',
                'T197', 'T114', 'T109', 'T121', 'T192', 'T127']
            if atk_embedding == "coder":
                vec_path = fpre + "/external/drug_word_vecs_umlsBERT.npz"
                # vec_path = fpre + "/external/drug_word_vecs_umlsBERT.npz"
            elif atk_embedding == "gte":
                vec_path = fpre + "/external/drug_word_vecs_gtebase.npz"
        
        elif type == "diseases":
            df = pd.read_csv(fpre + "/external/CTD_unique_disease_names_cleaned.csv")
            tui = ['T020', 'T190', 'T049', 'T019', 'T047', 'T050', 'T033',
            'T037', 'T048', 'T191', 'T046', 'T184']
            # vec_path = fpre + "/external/disease_word_cleaned_vecs_umlsBERT.npz"
            if atk_embedding == "coder":
                vec_path = fpre + "/external/disease_word_cleaned_vecs_umlsBERT.npz"
            elif atk_embedding == "gte":
                vec_path = fpre + "/external/disease_word_cleaned_vecs_gtebase.npz"
        else:
            raise NotImplementedError
        names = set(df["name"])
        # names = np.load("../qstab/external/all_drug_word_vecs_umlsBERT.npz", allow_pickle=True)['names']
        # names = np.squeeze(names)
        
        # Load text annotations
        if qa_dataset == "usmle":
            json_path = fpre + "/external/medqa_usmle_train_choices_annotation.json"
        elif qa_dataset == "medmcqa":
            json_path = fpre + "/external/medmcqa_train_leaner_choices_annotation.json"
        
        with open(json_path) as f:
            json_entities = json.load(f)
        constraints = [AnswerConstraint(df_dataset, json_entities, tui), 
                                     NamedEntityConstraint(False, entity_set=names),
                                     VectorScore(0.6, df, vec_path)]

        if atk_embedding == "coder":
            attack_embeddings = CODER_embedding(n_neighbors=500, ent_type=type)
        elif atk_embedding == "gte":
            attack_embeddings = GTE_embedding(n_neighbors=500, ent_type=type)
        elif atk_embedding == "cglove":
            attack_embeddings = counterfitted_GLOVE_embedding()
        else:
            raise NotImplementedError

    else:
        attack_embeddings = counterfitted_GLOVE_embedding()
    if "lr" in kwargs:
        lr = kwargs["lr"]
    else:
        lr = 0.1
    transformation = WordSwapEmbeddingDisplacement(
        max_candidates=candidates, embedding=attack_embeddings, learning_rate=lr, discretize_by_cosine=True)

    pre_transformation = [c for c in constraints if isinstance(c, PreTransformationConstraint)]
    
    goal_function = ZOOUntargetedClassification(model)
    kwargs.pop("lr", None)
    print("cand_select = {}, expn = {}".format(cand_select, expn))
    search_method = DiscreteZOO(candidates=candidates,
                                max_changes_per_word=grad_steps,
                                word_embeddings=attack_embeddings,
                                normalize_differences=True,
                                neighborhood_multiplier=5,
                                sample_cos_nn=True,
                                threshold_value=0.9,
                                threshold_samples=False,
                                short_circuit=True,
                                pre_transformation_constraints = pre_transformation,
                                transformation = transformation,
                                wir_method="random",
                                cand_select=cand_select,
                                expn=expn, **kwargs)
    # print("CONSTRAINTS: ", constraints)
    #constraints = []
    return Atk(goal_function, constraints, transformation,
                                    search_method)
