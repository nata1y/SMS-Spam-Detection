import pandas as pd
import numpy as np

from datadrift_detect.detect_alibi import _detect_drift
from deploy_model.feed_data_artificially import get_loss_and_nlp

class DriftManager:

  window_size = 100

  calls: int
  data: list
  stats: list
  incoming_real_labels: list
  preprocessed: list

  def __init__(self) -> None:
      self.name = "Manager 1"
      self.calls = 0
      self.data = np.array([])
      self.stats = np.array([])
      self.preprocessed = np.array([])
      self.incoming_real_labels = pd.read_csv(
          'regression_dataset/SMSSpamCollection_diff',
          sep='\t',
          names=['label', 'message']
      )
      

  def add_call(self, prediction, stats, preprocessed):
    self.calls = self.calls + 1
    self.stats = stats
    if len(self.preprocessed) == 0:
      self.preprocessed = np.array([preprocessed])
    else:
      self.preprocessed = np.append(self.preprocessed, [preprocessed], axis=0)
    if len(self.data) == 0:
      self.data = np.array([prediction])
    else:
      self.data = np.append(self.data, [prediction], axis=0)
      
      if len(self.data) % self.window_size == 0:
        self.calculate_drifts()

  def calculate_drifts(self):
    print("Checking last 10 elements for data drift...")
    indices = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    last_10 = pd.DataFrame(np.take(self.data, indices, axis=0), columns=['label', 'message'])
    _detect_drift(last_10)

    print("Checking complete incoming dataset for data drift...")
    full_set = pd.DataFrame(np.array(self.data), columns=['label', 'message'])
    _detect_drift(full_set)

    print("Check for concept drift using NLP and loss distribution")
    # get_loss_and_nlp(self.incoming_real_labels, last_10, self.stats)
    print("Check for full dataset")
    get_loss_and_nlp(self.incoming_real_labels, full_set, self.stats)
