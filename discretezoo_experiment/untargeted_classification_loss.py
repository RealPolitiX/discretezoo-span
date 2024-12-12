"""

Determine successful in untargeted Classification
----------------------------------------------------
"""

import numpy as np
import torch

# from textattack.goal_functions import ClassificationGoalFunction
from .goal_functions import ClassificationGoalFunction

class ZOOUntargetedClassification(ClassificationGoalFunction):
  """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

  def __init__(self,
               *args,
               target_max_score=None,
               kappa=0.0,
               interpolation=1.0,
               semantic_constraint=None,
               linguistic_constraint=None,
               **kwargs):
    self.target_max_score = target_max_score
    self.kappa = kappa
    self.interpolation = interpolation
    self.semantic_constraint = semantic_constraint
    self.linguistic_constraint = linguistic_constraint
    super().__init__(*args, **kwargs)

  def _is_goal_complete(self, model_output, _):
    if self.target_max_score:
      return model_output[self.ground_truth_output] < self.target_max_score
    elif (model_output.numel() == 1) and isinstance(self.ground_truth_output,
                                                    float):
      return abs(self.ground_truth_output -
                 model_output.item()) >= (self.target_max_score or 0.5)
    else:
      return model_output.argmax() != self.ground_truth_output

  def _get_score(self, model_output, attacked_text):
    # If the model outputs a single number and the ground truth output is
    # a float, we assume that this is a regression task.
    if (model_output.numel() == 1) and isinstance(self.ground_truth_output,
                                                  float):
      return abs(model_output.item() - self.ground_truth_output)
    else:
      indices = list(range(model_output.shape[0]))
      del indices[self.ground_truth_output]
      other_class_probs = model_output[indices]
      max_other_probs = max(other_class_probs)
      # This becomes negative because we will maximize it.
      loss = -np.maximum(
          np.log(model_output[self.ground_truth_output]) -
          np.log(max_other_probs) + self.kappa, 0)
      # print(loss)

      if self.semantic_constraint is not None:
        self.initial_attacked_text.attack_attrs["newly_modified_indices"] = {0}
        semantic_similarity_score = self.semantic_constraint._sim_score(
            self.initial_attacked_text, attacked_text)
        loss += self.interpolation * semantic_similarity_score
        del self.initial_attacked_text.attack_attrs["newly_modified_indices"]

    return loss