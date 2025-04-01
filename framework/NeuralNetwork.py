import numpy as np

class ActivationFunction:
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
