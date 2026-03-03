from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from fake_news.utils.features import Datapoint


class Model(ABC):
    @abstractmethod
    def train(self,
              train_datapoints: List[Datapoint],
              val_datapoints: List[Datapoint],
              cache_featurizer: Optional[bool] = False) -> None:

        pass
    
    @abstractmethod
    def predict(self, datapoints: List[Datapoint]) -> np.array:

        pass
    
    @abstractmethod
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        """
        Compute a set of model-specifc metrics on the provided set of datapoints.
        :param eval_datapoints: Datapoints to compute metrics for
        :param split: Data split on which metrics are being computed
        :return: A dictionary mapping from the name of the metric to its value
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return the model-specific parameters such as number of hidden-units in the case
        of a neural network or number of trees for a random forest
        :return: Dictionary containing the model parameters
        """
        pass
