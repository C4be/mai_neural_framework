# Нейросетевой фреймворк для машинного обучения

## Описание

### Обзор архитектуры
Библиотека предоставляет полный стек для создания и обучения нейронных сетей. Основные компоненты:

#### 1. DataSetLoader 🗂️
**Назначение:** Работа с данными

| Метод                | Параметры               | Возвращает | Описание                                                                 |
|-----------------------|-------------------------|------------|-------------------------------------------------------------------------|
| `__init__`            | Название датасета       | -          | Автоматическая загрузка iris/mnist/diabetes                             |
| `load_dataset`        | Название датасета       | -          | Загрузка данных в память                                                |
| `get_stats`           | -                       | Словарь    | Возвращает средние значения и стандартные отклонения                    |
| `head`/`tail`        | Количество образцов     | Кортеж     | Первые/последние n примеров данных                                      |
| `info_small`          | -                       | -          | Краткая сводка: размерности и пример данных                             |
| `info_full`           | -                       | -          | Полная статистика с распределением меток и описанием признаков          |
| `shuffle`             | -                       | -          | Перемешивание данных с сохранением соответствия данных и меток          |
| `batch`               | Размер батча            | Генератор  | Итеративная выдача данных порциями                                      |
| `map`                 | Функция преобразования  | -          | Применение функции ко всем данным                                       |
| `filter`              | Функция-фильтр          | -          | Удаление образцов не удовлетворяющих условию                           |
| `split`               | Размер тестовой выборки | Кортеж     | Разделение на train/test                                                |
| `standard_scaler`     | -                       | -          | Нормализация данных по z-оценке                                         |

---

### 2. ActivationFunction ⚡
**Реализованные функции:**

```python
# ReLU
forward(x) = max(0, x) 
backward(x) = 1 если x > 0, иначе 0

# Sigmoid
forward(x) = 1 / (1 + e^(-x))
backward(x) = σ(x) * (1 - σ(x))

# Tanh
forward(x) = (e^x - e^(-x)) / (e^x + e^(-x))
backward(x) = 1 - tanh²(x)

# LeakyReLU
forward(x) = x (x > 0), 0.01x (x <= 0)
backward(x) = 1 (x > 0), 0.01 (x <= 0)
```

---

### 3. Optimizer 🎯
**Доступные алгоритмы:**

| Оптимизатор    | Параметры                          | Формула обновления                     |
|---------------|-----------------------------------|----------------------------------------|
| SGD           | `learning_rate`                  | `w = w - η*∇`                          |
| MomentumSGD    | `learning_rate`, `momentum`       | `v = γ*v + η*∇`, `w = w - v`           |
| RMSprop        | `learning_rate`, `decay_rate`     | `E[g²] = ρE[g²] + (1-ρ)g²`, `w = w - η*g/√(E[g²]+ε)` |
| Adam           | `learning_rate`, `beta1`, `beta2` | Комбинация Momentum и RMSprop с bias correction |

---

### 4. Layer 🧱
**Архитектура:**
```python
class Layer:
    def __init__(input_size, output_size, activation):
        self.weights = инициализация xavier
        self.bias = нулевой вектор
    
    # Прямой проход
    def forward(inputs) → outputs
    
    # Обратное распространение
    def backward(grad_output) → grad_input
```

---

### 5. NeuralNetwork 🧠
**Рабочий цикл:**
1. Инициализация сети с оптимизатором
```python
model = NeuralNetwork(Optimizer.Adam(lr=0.001))
```
2. Добавление слоев
```python
model.add_layer(Layer(4, 10, ActivationFunction.ReLU))
model.add_layer(Layer(10, 3, ActivationFunction.Softmax))
```
3. Обучение
```python
model.fit(X_train, y_train, epochs=100)
```
4. Предсказание
```python
predictions = model.predict(X_test)
```

---

## Ключевые особенности
- **Гибкость**: Поддержка кастомных слоев и активаций
- **Производительность**: Векторизованные вычисления через numpy
- **Аналитика**: Встроенные методы анализа данных
- **Модульность**: Независимое использование компонентов

## Пример использования
```python
# Инициализация датасета
dl = DataSetLoader('iris')
dl.shuffle()
X_train, X_test, y_train, y_test = dl.split(0.2)

# Создание модели
model = NeuralNetwork(Optimizer.Adam(lr=0.01))
model.add_layer(Layer(4, 10, ActivationFunction.ReLU))
model.add_layer(Layer(10, 3, ActivationFunction.Sigmoid))

# Обучение
loss_history = model.fit(X_train, y_train, epochs=100)

# Оценка
accuracy = model.evaluate(X_test, y_test)
print(f"Точность модели: {accuracy:.2%}")
```
```