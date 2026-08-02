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

        pass
    
    @abstractmethod
    def get_params(self) -> Dict:

        pass