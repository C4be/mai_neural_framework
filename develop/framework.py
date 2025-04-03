import numpy as np
import requests
from sklearn import datasets
import pandas as pd
from typing import Tuple, List, Callable, Any, Dict

####################################  START DataSetLoader ####################################
class DataSetLoader:
    
    __datasets = ['iris', 'mnist', 'diabetes']
    
    #========================= Загрузка датасета ================================
    
    def __init__(self, dname: str = ''):
        "Инициализация датасета"
        self.name: str = ''
        self.data = None
        self.labels = None
        
        # Статус загрузки
        self.isLoad = False
        
        if dname in self.__datasets:
            self.name = dname
            self.load_dataset(dname)
            self.isLoad = True
        else:
            raise ValueError(f"Датасет {dname} не найден")
        
    def load_dataset(self, dname: str = ''):
        "Загрузка датасета"
        if dname == 'iris':
            self.data, self.labels = datasets.load_iris(return_X_y=True)
        elif dname == 'mnist':
            self.data, self.labels = datasets.load_digits(return_X_y=True)
        elif dname == 'diabetes':
            self.data, self.labels = datasets.load_diabetes(return_X_y=True)
        else:
            raise ValueError(f"Датасет {dname} не найден")
        
    def get_available_datasets(self) -> List[str]:
        "Получение списка доступных датасетов"
        return self.__datasets
    
    #========================= Информация ================================
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """Calculate and return dataset statistics"""
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        return {
            'mean': mean,
            'std': std
        }

    def head(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Первые n элементов датасета"""
        return self.data[:n], self.labels[:n]

    def tail(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Последние n элементов датасета"""
        return self.data[-n:], self.labels[-n:]
    
    def info_small(self) -> None:
        """Информация о датасете"""
        print(f'Текущий датасет: {self.name}')
        print(f'Размерность: {self.data.shape}')
        print(f'Метки: {self.labels.shape}')
        print(f'Пример данных: {self.head()}')

    def info_full(self) -> None:
        """Полная информация о датасете"""
        stats = self.get_stats()
        print(f'Текущий датасет: {self.name}')
        print(f'Размерность: {self.data.shape}')
        print(f'Метки: {self.labels.shape}')
        print(f'Пример данных: {self.head() + self.tail()}')
        print(f'Статистика: \n{pd.DataFrame(self.data).describe()}')
        print(f'Количество меток: \n{pd.Series(self.labels).value_counts()}')
        print(f'Среднее значение признаков: \n{stats["mean"]}')
        print(f'Стандартное отклонение признаков: \n{stats["std"]}')
        
    #========================= Обработка ================================

    def shuffle(self) -> None:
        """Перемешивание данных"""
        idx = np.random.permutation(len(self.data))
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        
    def batch(self, data_tuple: Tuple[np.ndarray, np.ndarray], batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into batches. Can accept train or test sets from split() function"""
        data, labels = data_tuple
        
        if shuffle:
            # Create shuffled indices
            idx = np.random.permutation(len(data))
            data = data[idx]
            labels = labels[idx]
            
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size], labels[i:i+batch_size]
            
    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Применение функции к данным"""
        self.data = func(self.data)

    def filter(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Фильтрация данных"""
        self.data = self.data[func(self.data)]
        
    def split(self, test_size: float = 0.2, shuffle: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Разделение данных на обучающую и тестовую выборки"""
        if shuffle:
            self.shuffle()
            
        split_idx = int(len(self.data) * (1 - test_size))
        return (self.data[:split_idx], self.labels[:split_idx]), (self.data[split_idx:], self.labels[split_idx:])
    
    def standard_scaler(self) -> None:
        """
        Стандартизирует признаки путем удаления среднего значения и масштабирования до единичной дисперсии.
        Формула: z = (x - среднее) / стандартное_отклонение
        """
        stats = self.get_stats()
        std = np.where(stats['std'] == 0, 1, stats['std'])
        self.data = (self.data - stats['mean']) / std
####################################  END DataSetLoader ####################################


####################################  START ActivationFunction ####################################
class ActivationFunction:
    class Linear:
        @staticmethod
        def forward(x):
            return x
        
        @staticmethod
        def backward(x):
            return np.ones_like(x)
        
    class ReLU:
        @staticmethod
        def forward(x):
            return np.maximum(0, x)
        
        @staticmethod
        def backward(x):
            return np.where(x > 0, 1, 0)
    
    class Sigmoid:
        @staticmethod
        def forward(x):
            return 1 / (1 + np.exp(-x))
        
        @staticmethod
        def backward(x):
            s = ActivationFunction.Sigmoid.forward(x)
            return s * (1 - s)
    
    class Tanh:
        @staticmethod
        def forward(x):
            return np.tanh(x)
        
        @staticmethod
        def backward(x):
            return 1 - np.tanh(x)**2
    
    class LeakyReLU:
        @staticmethod
        def forward(x, alpha=0.01):
            return np.where(x > 0, x, alpha * x)
        
        @staticmethod
        def backward(x, alpha=0.01):
            return np.where(x > 0, 1, alpha)
    
    class Softmax:
        @staticmethod
        def forward(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
        
        @staticmethod
        def backward(x):
            # Gradient for cross-entropy + softmax combination
            return np.ones_like(x)

####################################  END ActivationFunction ####################################

####################################  START LossFunctions ####################################
class LossFunctions:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)])
        return np.sum(log_probs) / m
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]

    # Добавляем MSE loss и его производную
    @staticmethod
    def mse(y_pred, y_true):
        """Вычисляет среднеквадратичную ошибку"""
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def mse_derivative(y_pred, y_true):
        """Производная MSE по предсказаниям"""
        return 2 * (y_pred - y_true) / y_true.size

####################################  END LossFunctions ####################################

####################################  START Optimizer ####################################
class Optimizer:
    class SGD:
        def __init__(self, learning_rate=0.01):
            self.learning_rate = learning_rate
        
        def update(self, params, grads):
            for param, grad in zip(params, grads):
                param -= self.learning_rate * grad
    
    class MomentumSGD:
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.velocities = None
        
        def update(self, params, grads):
            if self.velocities is None:
                self.velocities = [np.zeros_like(param) for param in params]
            
            for param, grad, velocity in zip(params, grads, self.velocities):
                velocity = self.momentum * velocity - self.learning_rate * grad
                param += velocity
    
    class RMSprop:
        def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.decay_rate = decay_rate
            self.epsilon = epsilon
            self.cache = None
        
        def update(self, params, grads):
            if self.cache is None:
                self.cache = [np.zeros_like(param) for param in params]
            
            for param, grad, cache in zip(params, grads, self.cache):
                cache = self.decay_rate * cache + (1 - self.decay_rate) * grad**2
                param -= self.learning_rate * grad / (np.sqrt(cache) + self.epsilon)
    
    class Adam:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = None
            self.v = None
            self.t = 0
        
        def update(self, params, grads):
            if self.m is None:
                self.m = [np.zeros_like(param) for param in params]
                self.v = [np.zeros_like(param) for param in params]
            
            self.t += 1
            
            for param, grad, m, v in zip(params, grads, self.m, self.v):
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad**2
                
                m_hat = m / (1 - self.beta1**self.t)
                v_hat = v / (1 - self.beta2**self.t)
                
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
####################################  END Optimizer ####################################

####################################  START Layer ####################################
class Layer:
    def __init__(self, input_size, output_size, activation_function=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation_function
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            self.activated_output = self.activation.forward(self.output)
            return self.activated_output
        return self.output
    
    def backward(self, grad_output):
        if self.activation:
            grad_output = grad_output * self.activation.backward(self.output)
        
        self.grad_weights = np.dot(self.inputs.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
####################################  END Layer ####################################

####################################  START NeuralNetwork ####################################
class NeuralNetwork:
    def __init__(self, optimizer):
        self.layers = []
        self.optimizer = optimizer
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def update(self):
        params = []
        grads = []
        for layer in self.layers:
            params.extend([layer.weights, layer.bias])
            grads.extend([layer.grad_weights, layer.grad_bias])
        self.optimizer.update(params, grads)
        
    def summary(self):
        """Информация по архитектуре нейросети"""
        print('Neural Network Summary:')
        print('=' * 80)
        print(f"{'Layer Type':<20} {'Output Shape':<20} {'Params':<15}")
        print('=' * 80)
        
        total_params = 0
        input_shape = None
        
        for i, layer in enumerate(self.layers):
            # Calculate layer parameters
            params = np.prod(layer.weights.shape) + layer.bias.size
            total_params += params
            
            # Get layer output shape
            if i == 0:
                input_shape = layer.weights.shape[0]
            output_shape = layer.weights.shape[1]
            
            # Format activation name
            activation = layer.activation.__class__.__name__ if layer.activation else "None"
            
            # Print layer info
            print(f"Dense {i+1:<14} ({input_shape}, {output_shape}){' '*5} {params:<15}")
            print(f"{'Activation: ' + activation:<20} {' '*20} {'0':<15}")
            
            input_shape = output_shape
            
        print('=' * 80)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {total_params}")
        print(f"Non-trainable params: 0")
        print('=' * 80)
####################################  END NeuralNetwork ####################################