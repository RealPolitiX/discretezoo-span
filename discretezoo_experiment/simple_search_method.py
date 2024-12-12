"""
Discrete ZOO search algorithm
==========================
"""
import utils as u
import random

import numpy as np
from copy import copy
import torch
from torch.nn.functional import softmax

from textattack.shared import AttackedText, utils
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from transformers import AutoModel, AutoTokenizer

from sklearn.utils import shuffle

tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

select_method = "random"

class DiscreteZOO(SearchMethod):
  """Reimplementation of the DiscreteZOO attack in the textattack framework.
  """

  def __init__(self,
               word_embeddings,
               candidates=10,
               neighborhood_multiplier=5,
               max_changes_per_word=1,
               max_changes_per_sentence=0,
               wir_method="unk",
               normalize_displacements=True,
               normalize_differences=False,
               average_displacements=False,
               sample_cos_nn=False,
               threshold_samples=False,
               threshold_value=0.9,
               min_neighbors=1,
               short_circuit=False,
               discretize_furthest=False,
               logging=False,
               pre_transformation_constraints = [],
               transformation = None,
               cand_select = "random",
               expn = 0,
               qtag = True):
    # print("DiscreteZOO, cand_select = {}, expn = {}".format(cand_select, expn))
    self.word_embeddings = word_embeddings
    self.candidates = candidates
    self.max_gradient_steps = max_changes_per_word
    self.max_changes_per_sentence = max_changes_per_sentence
    self.wir_method = wir_method
    self.normalize_displacements = normalize_displacements
    self.normalize_differences = normalize_differences
    self.average_displacements = average_displacements
    self.neighborhood_multiplier = neighborhood_multiplier
    self.sample_cos_nn = sample_cos_nn
    self.threshold_samples = threshold_samples
    self.threshold_value = threshold_value
    self.min_neighbors = min_neighbors
    self.short_circuit = short_circuit
    self.discretize_furthest = discretize_furthest
    self.logging = logging
    self.pre_transformation_constraints = pre_transformation_constraints
    self.transformation = transformation
    self.cand_select = cand_select
    self.expn = expn
    self.qtag = qtag # Add [Question] tag to question

  def extra_repr_keys(self):
    return [
        'candidates', 'max_gradient_steps', 'normalize_displacements',
        'normalize_differences', 'average_displacements',
        'neighborhood_multiplier', 'sample_cos_nn', 'threshold_samples',
        'threshold_value', 'min_neighbors',
    ]

  def _check_constraints(self, transformed_text, current_text, original_text):
    """Check if `transformted_text` still passes the constraints with
    respect to `current_text` and `original_text`.

    This method is required because of a lot population-based methods does their
    own transformations apart from the actual `transformation`. Examples include
    `crossover` from `GeneticAlgorithm` and `move` from
    `ParticleSwarmOptimization`.
    Args:
      transformed_text (AttackedText): Resulting text after transformation
      current_text (AttackedText): Recent text from which `transformed_text` was
        produced from.
      original_text (AttackedText): Original text
    Returns
      `True` if constraints satisfied and `False` if otherwise.
    """
    filtered = self.filter_transformations([transformed_text],
                                           current_text,
                                           original_text=original_text)
    return True if filtered else False

  def _get_index_order(self, initial_text):
    """Returns word indices of ``initial_text`` in descending order of
    importance."""
    len_text = len(initial_text.words)
    # precon = self.pre_transformation_constraints[0]
    # precon(initial_text, self.transformation)
    # len_text = len(precon._get_modifiable_indices(initial_text))

    if self.wir_method == "unk":
      leave_one_texts = [
          initial_text.replace_word_at_index(i, "[UNK]")
          for i in range(len_text)
      ]
      leave_one_results, search_over = self.get_goal_results(leave_one_texts)
      index_scores = np.array([result.score for result in leave_one_results])

    elif self.wir_method == "weighted-saliency":
      # first, compute word saliency
      leave_one_texts = [
          initial_text.replace_word_at_index(i, "[UNK]")
          for i in range(len_text)
      ]
      leave_one_results, search_over = self.get_goal_results(leave_one_texts)
      saliency_scores = np.array([result.score for result in leave_one_results])

      softmax_saliency_scores = softmax(torch.Tensor(saliency_scores),
                                        dim=0).numpy()

      # compute the largest change in score we can find by swapping each word
      delta_ps = []
      for idx in range(len_text):
        transformed_text_candidates = self.get_transformations(
            initial_text,
            original_text=initial_text,
            indices_to_modify=[idx],
        )
        if not transformed_text_candidates:
          # no valid synonym substitutions for this word
          delta_ps.append(0.0)
          continue
        swap_results, _ = self.get_goal_results(transformed_text_candidates)
        score_change = [result.score for result in swap_results]
        max_score_change = np.max(score_change)
        delta_ps.append(max_score_change)

      index_scores = softmax_saliency_scores * np.array(delta_ps)

    elif self.wir_method == "delete":
      leave_one_texts = [
          initial_text.delete_word_at_index(i) for i in range(len_text)
      ]
      leave_one_results, search_over = self.get_goal_results(leave_one_texts)
      index_scores = np.array([result.score for result in leave_one_results])

    elif self.wir_method == "random":
      index_order = np.arange(len_text)
      np.random.shuffle(index_order)
      search_over = False
    else:
      index_order = None
      raise ValueError(f"Unsupported WIR method {self.wir_method}")

    if self.wir_method != "random":
      index_order = (-index_scores).argsort()

    return index_order, search_over

  def _get_candidates(self, current_attack):
    """This samples tokens nearby in embedding space, ignoring constraints.

        In order for the algorithm to work, we want to sample tokens nearby without
        being constrained because even words that don't fit can still be informative
        when calculating displacements. self.get_transformations filters words
        after finding neighbors in embedding space.

        Args:
          current_attack: An AttackedText with our current in-progress attack.
          target_index: The index of the word we want to replace.
        """
    # indices_to_change = {target_index}
    # # print("Index to change is {}".format(indices_to_change))
    # # for constraint in self.pre_transformation_constraints:
    # #   indices_to_change = indices_to_change & constraint(
    # #       current_attack, self.transformation)
    # if len(indices_to_change) == 0: #Place where the outcome affects the realized attack budget
    #   # print("Nothing!\n")
    #   return []
    # token_to_change = current_attack.words[target_index]
    # embedding = self.word_embeddings[token_to_change] # retrieve the embedding vector for the word
    # print("{} is processed!\n".format(token_to_change))

    span_to_change = current_attack
    # print("Span to attack is {}".format(span_to_change))
    embedding = span2vec(span_to_change, model, tokenizer)
    # if embedding is None:
    #   embedding = self.word_embeddings[token_to_change.casefold()]
    # print(self.word_embeddings['Ceftriaxone'])
    if embedding is None:
      print("{} is nothing!\n".format(span_to_change))
      return []
    if self.sample_cos_nn:
      candidate_list, distance_list = self.word_embeddings.get_cos_nn(
          embedding, 1 + self.candidates * self.neighborhood_multiplier)
      candidate_list = candidate_list[1:]
      distance_list = distance_list[1:]
      if self.threshold_samples:
        candidate_list = [
            candidate
            for candidate, distance in zip(candidate_list, distance_list)
            if distance >= self.threshold_value
        ]
    else:
      candidate_list, distance_list = self.word_embeddings.get_euc_nn(
          embedding, 1 + self.candidates * self.neighborhood_multiplier)
      candidate_list = candidate_list[1:]
      distance_list = distance_list[1:]
      if self.threshold_samples:
        candidate_list = [
            candidate
            for candidate, distance in zip(candidate_list, distance_list)
            if distance <= self.threshold_value
        ]
    #utils.logger.info("There are " + str(len(candidate_list)) + " acceptable replacement tokens.")
    # print("Number of substitution candidates is {}".format(len(candidate_list)))
    if len(candidate_list) < self.min_neighbors:
      return []
    if len(candidate_list) < self.candidates:
      return candidate_list

    if self.cand_select == "random":
        random.seed(42)
        candidate_list = random.sample(candidate_list, k=self.candidates)
        # candidate_list = np.array(candidate_list)
        # candidat_list = np.random.choice(candidate_list, size=self.candidates, replace=False)
        # candidate_list = candidate_list.tolist()
    elif self.cand_select == "closest":
        # Sample the closest few members
        distance_list = np.array(distance_list)
        candidate_list = np.array(candidate_list)
        candidate_list = candidate_list[np.argsort(distance_list)][:self.candidates]
        candidate_list = candidate_list.tolist()
    elif self.cand_select == "pdws":
        n_pdws = self.expn
        print("Exponent is {}".format(n_pdws))
        # Powerscaled distance-weighted sampling
        np.random.seed(42)
        distance_list = np.array(distance_list)
        candidate_list = np.array(candidate_list)
        probs = distance_list**(n_pdws)/np.sum(distance_list**(n_pdws))
        final_list = np.random.choice(candidate_list, size=self.candidates, p=probs, replace=False)
        candidate_list = final_list.tolist()
    
    return candidate_list

  def _transform_text(self, previous_result, original_text, target_id,
                      target_text, target_boundary, **kwargs):
    """Creates possible texts given our current attack.

    Args:
      current_attack: An AttackedText with our current in-progress attack.
      original_result: The original text result, GoalFunctionResult.
      target_index: The index of the word that we are currently attacking.
    """
    search_over = False
    updated_result = previous_result
    for i in range(self.max_gradient_steps):
      current_attack = previous_result.attacked_text
      candidate_tokens = self._get_candidates(target_text)
      # print(candidate_tokens)

      if candidate_tokens == []:
        return previous_result, search_over

      # candidates = [
      #     current_attack.replace_word_at_index(target_index, token)
      #     for token in candidate_tokens
      # ]
      # print(candidate_tokens)
      candidates = [entity_perturb(original_text, target_id, target_text, target_boundary, qtag=True)
          for target_text in candidate_tokens]

    #   current_embedding = (
    #       self.word_embeddings[current_attack.words[target_index]])
    #   print(current_attack.words)
      # current_embedding = (span2vec(current_attack.words[target_index], model, tokenizer))
      # current_embedding = (span2vec(current_attack.text, model, tokenizer))
      current_embedding = (span2vec(target_text, model, tokenizer))
      #remove
      #changed_tokens = []
      #for candidate in candidates:
      #  changed_tokens.append(candidate.words[target_index])

      new_results, search_over = self.get_goal_results(candidates)
      if self.short_circuit:
        for result in new_results:
          if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            if self.logging:
              print("Success by short circuiting")
            return result, True
      
      # spans_to_change = []
      # changed_tokens_embeddings = np.array(
      #     [self.word_embeddings[token] for token in candidate_tokens])
      # changed_tokens_embeddings = np.array(
      #     [span2vec(span, model, tokenizer) for span in candidate_tokens])
      changed_tokens_embeddings = np.array(
          [span2vec(span, model, tokenizer) for span in candidate_tokens])
      
      displacement_vectors = changed_tokens_embeddings - current_embedding

      eps = np.finfo(displacement_vectors.dtype).eps
      normalizers = np.maximum(np.linalg.norm(displacement_vectors, axis=-1), eps)
      
      if self.logging:
        print("Mean Displacement Norm = " + str(np.mean(normalizers)))

      new_result_scores = np.array([result.score for result in new_results])
      result_score_diffs = np.expand_dims(
          new_result_scores - previous_result.score, 1)

      if self.normalize_displacements:
        displacement_vectors = displacement_vectors / np.expand_dims(
            normalizers, 1)

      if self.normalize_differences:
        if result_score_diffs.shape != normalizers.shape:
          return previous_result, search_over
        result_score_diffs = result_score_diffs / np.expand_dims(normalizers, 1)

      weighted_displacements = displacement_vectors * result_score_diffs

      if self.average_displacements:
        reduced_displacement = np.mean(weighted_displacements, axis=0)
      else:
        reduced_displacement = np.sum(weighted_displacements, axis=0)

      if self.logging:
        print("Reduced Displacement Norm" + str(np.linalg.norm(reduced_displacement)))

      self.transformation.displacement = reduced_displacement
      new_candidates = self.get_transformations(
          current_attack,
          original_text,
          indices_to_modify=[target_index])
      # new_candidates = []

      if len(new_candidates) == 0:
        return previous_result, search_over
      else:
        #The 0th candidate in the list is the closest in the list to the displacement + embedding
        updated_candidate = new_candidates[0]
        updated_results, search_over = self.get_goal_results(
            [updated_candidate])
        if len(updated_results) == 0:
          return previous_result, search_over
        updated_result = updated_results[0]
        if updated_result.goal_status == GoalFunctionResultStatus.SUCCEEDED or search_over:
          return updated_result, search_over
        previous_result = updated_result
    return previous_result, search_over

  def perform_search(self, initial_result):
    """Initializes a population and perturbs them to search for attacks
    Args:
      initial_result (GoalFunctionResult): Original text result

    """
    # attack_order = self._get_index_order(initial_result.attacked_text)[0]

    # number_of_target_tokens = (self.max_changes_per_sentence
    #                            if self.max_changes_per_sentence != 0 else len(
    #                                initial_result.attacked_text.words))
    original_text = initial_result.attacked_text
    current_result = initial_result
    # print("Goal status is {}, succeed state is {}".format(current_result.goal_status, GoalFunctionResultStatus.SUCCEEDED))
    # attack_order = attack_order[:number_of_target_tokens]

    # Attack with a randomly shuffled attack order
    # precon = self.pre_transformation_constraints[0]
    # attack_order = precon(initial_result.attacked_text, self.transformation)
    # attack_order = np.array(list(attack_order))
    # np.random.shuffle(attack_order)
    # attack_order = set(attack_order.tolist())

    precon = self.pre_transformation_constraints[0]
    attack_spans, attack_optids, attack_boundaries = precon(initial_result.attacked_text, self.transformation)
    
    # attack_spans = np.array(attack_spans)
    # attack_spans, attack_boundaries = shuffle(attack_spans, attack_boundaries, random_state=42)
    # attack_spans = attack_spans.tolist()

    # print(attack_order)

    # for target_index in attack_order:
    #   updated_result, search_over = self._transform_text(
    #       current_result, original_text, target_index)
    #   if (search_over or
    #       updated_result.goal_status == GoalFunctionResultStatus.SUCCEEDED):
    #     # print("Updated!")
    #     return updated_result

    for target_text, target_id, target_boundary in zip(attack_spans, attack_optids, attack_boundaries):
      
      # updated_result, search_over = entity_perturb(original_text, target_id,
      #                                              target_text, target_boundary)
      updated_result, search_over = self._transform_text(
          current_result, original_text, target_id, target_text, target_boundary)
      if (search_over or
          updated_result.goal_status == GoalFunctionResultStatus.SUCCEEDED):
        return updated_result
      
      if self.short_circuit and (updated_result.score < current_result.score):
        if self.logging:
          print("avoided moving to bad token by short circuiting")
        continue
      current_result = updated_result
    # print("Goal status is {}, succeed state is {}".format(current_result.goal_status, GoalFunctionResultStatus.SUCCEEDED))
    
    return current_result

  @property
  def is_black_box(self):
    """Returns `True` if search method does not require access to victim
        model's internal states."""
    return True


def span2vec(span, model, tokenizer):
    
    span_info = tokenizer(span, return_tensors='pt')
    span_ids = span_info['input_ids']
    span_mask = span_info['attention_mask']
    
    with torch.no_grad():
        vec = model(span_ids).last_hidden_state[0][1]
    
    return vec.numpy()


def entity_perturb(original_text, option_idx, replacement, boundary, qtag):
    
    text_before = original_text.text
    rfm = u.Reformatter(text_before)
    opt_text = rfm.options[option_idx]
    replacer = copy(replacement)
    # print(replacer)

    # opt_text = copy(original_option)
    a, b = boundary
    # print("Full option is {}, boundary is {}-{}".format(opt_text, a, b))
    span_unprtrb = opt_text[a:b]
    # print("Unperturbed span is {}".format(span_unprtrb))

    # Adjust for capitalization of the first letter
    cap_flag = span_unprtrb[0].isupper()
    if cap_flag:
        replacer = replacer.capitalize()
    else:
        replacer = replacer.lower()

    option_prtrb = opt_text[:a] + replacer + opt_text[b:]

    # Update the option to the perturbed version
    rfm.options[option_idx] = option_prtrb

    # Reformat text to the original form
    print(qtag)
    text_after = rfm.reformat(qtag=False)

    perturbed_text = AttackedText(text_after)
    
    # print('{} | {}'.format(option_prtrb, opt_text))
    return perturbed_text


# def entity_perturb(option, ent_tags, perturb_types, replacer):
#     ''' Entity perturbation.
    
#     option: Option for the question.
#     ent_tags: Annotations for the question options.
#     perturb_types: Compatible type unique identifiers (TUIs) for perturbation.
#     replacer: Entry to replace the existing part of text.
#     '''
    
#     tags = ent_tags['labels']
    
#     # Select an entity based on the type unique identifiers provided
#     # Check if some of the entity tags are compatible with the given perturbation types
#     type_flags = []
#     for tag in tags:
#         if tag in perturb_types:
#             type_flags.append(True)
#         else:
#             type_flags.append(False)
#     type_flags = np.array(type_flags)
#     n_passables = type_flags.sum()
    
#     # Return unperturbed options if no entity type matches
#     if n_passables == 0:
#         return option
#     # Return singly perturbed options when only one entity type matches
#     elif n_passables == 1:
#         pass_id = np.argwhere(type_flags == True).item()
#     # Return perturbed options after randomly select one type-matched entity to perturb
#     elif n_passables > 1:
#         passes = np.argwhere(type_flags == True).ravel()
#         pass_id = np.random.choice(passes, 1).item()
        
#     opt_text = copy(option)
#     a, b = ent_tags['boundaries'][pass_id]
#     span_unprtrb = opt_text[a:b]
    
#     cap_flag = span_unprtrb[0].isupper()
#     if cap_flag:
#         replacer = replacer.capitalize()
#     else:
#         replacer = replacer.lower()

#     option_prtrb = opt_text[:a] + replacer + opt_text[b:]
    
#     # print('{} | {}'.format(option_prtrb, opt_text))
#     return option_prtrb, span_unprtrb