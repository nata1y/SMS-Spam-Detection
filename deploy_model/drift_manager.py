from numpy.core.numeric import indices
import pandas as pd
import numpy as np
from datadrift_detect.detect_alibi import _detect_drift

class DriftManager:

  window_size = 100

  calls: int
  data: list

  def __init__(self) -> None:
      self.name = "Manager 1"
      self.calls = 0
      self.data = np.array([])
      

  def add_call(self, prediction):
    self.calls = self.calls + 1
    if len(self.data) == 0:
      self.data = np.array([prediction])
    else:
      self.data = np.append(self.data, [prediction], axis=0)
      
      if len(self.data) % self.window_size == 0:
        self.calculate_drifts()

  def calculate_drifts(self):
    print("Checking last 10 elements for data drift...")
    indices = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    df = pd.DataFrame(np.take(self.data, indices, axis=0), columns=['label', 'message'])
    _detect_drift(df)

    print("Checking complete incoming dataset...")
    df = pd.DataFrame(np.array(self.data), columns=['label', 'message'])
    _detect_drift(df)

  


  