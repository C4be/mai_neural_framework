import numpy as np
import requests
from sklearn import datasets
import pandas as pd
from typing import Tuple, List, Callable, Any


class DataLoader:

    __AvailableDatasets = ["iris", "mnist", "diabetes", "wine"]

    def __init__(self, dataset_name: str = None):``
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
                # MNIST - изображения рукописных цифр
                mnist_url = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
                response = requests.get(mnist_url)
                with open("mnist.pkl.gz", "wb") as f:
                    f.write(response.content)
                data = np.load("mnist.pkl.gz", allow_pickle=True)
                self.data = data[0][0]
                self.labels = data[0][1]
            elif dataset_name.lower() == "iris":
                # Iris - набор данных о цветках ириса
                iris = datasets.load_iris()
                self.data = iris.data
                self.labels = iris.target
            elif dataset_name.lower() == "diabetes":
                # Diabetes - набор данных о диабете
                diabetes = datasets.load_diabetes()
                self.data = diabetes.data
                self.labels = diabetes.target
            elif dataset_name.lower() == "wine":
                # Wine - набор данных о винах
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

    def split_dataset(
        self, test_size: float = 0.2, shuffle: bool = True
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split dataset into training and test sets

        Args:
            test_size: Proportion of the dataset to include in the test split (0 to 1)
            shuffle: Whether to shuffle the data before splitting

        Returns:
            ((X_train, y_train), (X_test, y_test)): Tuple containing train and test splits
        """
        if shuffle:
            self.shuffle()

        split_idx = int(len(self.data) * (1 - test_size))

        X_train = self.data[:split_idx]
        y_train = self.labels[:split_idx]
        X_test = self.data[split_idx:]
        y_test = self.labels[split_idx:]

        return (X_train, y_train), (X_test, y_test)
