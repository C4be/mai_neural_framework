import numpy as np
import requests
from sklearn import datasets
import pandas as pd
from typing import Tuple, List, Callable, Any


class DataLoader:
    
    __AvailableDatasets = ["iris", "mnist", "diabetes", "wine"]
    
    def __init__(self, dataset_name: str = None):
        """Инициализируем датасет, метки и если есть название, то загружаем его"""
        self.data = None
        self.labels = None
        if dataset_name:
            self.load_dataset(dataset_name)

    def list_datasets(self) -> List[str]:
        """Список доступных датасетов, которые можно загрузить с помощью метода load_dataset()"""
        return self.__AvailableDatasets
    
    def load_dataset(self, dataset_name: str) -> None:
        """Загрузка подготовленных датасетов"""
        try:
            if dataset_name.lower() == "mnist":
                # Load MNIST from remote URL since it's not in sklearn
                mnist_url = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
                response = requests.get(mnist_url)
                with open("mnist.pkl.gz", "wb") as f:
                    f.write(response.content)
                data = np.load("mnist.pkl.gz", allow_pickle=True)
                self.data = data[0][0]
                self.labels = data[0][1]
            elif dataset_name.lower() == "iris":
                # Load Iris dataset from sklearn
                iris = datasets.load_iris()
                self.data = iris.data
                self.labels = iris.target
            elif dataset_name.lower() == "diabetes":
                # Load Diabetes dataset from sklearn
                diabetes = datasets.load_diabetes()
                self.data = diabetes.data
                self.labels = diabetes.target
            elif dataset_name.lower() == "wine":
                # Load Wine dataset from sklearn
                wine = datasets.load_wine()
                self.data = wine.data
                self.labels = wine.target
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")
        except Exception as e:
            if isinstance(e, requests.RequestException):
                raise ConnectionError(
                    f"Failed to download dataset {dataset_name} from remote source"
                )
            raise e

    def head(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Get first n elements of the dataset"""
        return self.data[:n], self.labels[:n]

    def tail(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Get last n elements of the dataset"""
        return self.data[-n:], self.labels[-n:]

    def get_stats(self) -> dict:
        """Get dataset statistics"""
        stats = {
            "data_shape": self.data.shape,
            "labels_shape": self.labels.shape,
            "data_type": self.data.dtype,
            "labels_type": self.labels.dtype,
            "unique_labels": np.unique(self.labels),
            "data_mean": np.mean(self.data),
            "data_std": np.std(self.data),
        }
        return stats

    def batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split dataset into batches"""
        n_samples = len(self.data)
        indices = list(range(0, n_samples, batch_size))
        batches = [
            (self.data[i : i + batch_size], self.labels[i : i + batch_size])
            for i in indices
        ]
        return batches

    def shuffle(self) -> None:
        """Shuffle the dataset"""
        permutation = np.random.permutation(len(self.data))
        self.data = self.data[permutation]
        self.labels = self.labels[permutation]

    def map(self, func: Callable[[Any], Any]) -> None:
        """Apply a function to all data samples"""
        self.data = np.array([func(x) for x in self.data])
