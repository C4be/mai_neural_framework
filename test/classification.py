import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Добавляем путь к корневой директории проекта
sys.path.append("/Users/cube/Documents/Development/MAI/Сошников")

# Импортируем наши классы
from framework.DataLoader import DataLoader
from framework.NeuralNetwork import NeuralNetwork, Layer, ActivationFunction, Optimizer

def normalize_feature(x):
    return (x - np.mean(x)) / np.std(x)

def main():
    # Загружаем данные
    data_loader = DataLoader("iris")
    
    # Перемешиваем данные
    data_loader.shuffle()
    data_loader.map(normalize_feature)
    
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(data_loader.data, data_loader.labels, test_size=0.2, random_state=42)
    
    # Нормализуем данные
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Определяем архитектуру нейронной сети
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = len(np.unique(y_train))
    
    # Создаем оптимизатор
    optimizer = Optimizer.Adam(learning_rate=0.01)
    
    # Создаем нейронную сеть
    model = NeuralNetwork(optimizer)
    model.add_layer(Layer(input_size, hidden_size, ActivationFunction.ReLU))
    model.add_layer(Layer(hidden_size, output_size, ActivationFunction.Sigmoid))
    
    # Обучаем модель
    epochs = 100
    batch_size = 16
    # Добавить списки для хранения метрик
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        # Получаем батчи данных
        batches = data_loader.batch(batch_size)
        for X_batch, y_batch in batches:
            # Прямой проход
            y_pred = model.forward(X_batch)
            
            # Вычисляем ошибку (кросс-энтропия для классификации)
            y_one_hot = np.zeros((y_batch.size, y_pred.shape[1]))
            y_one_hot[np.arange(y_batch.size), y_batch.astype(int)] = 1
            
            loss = -np.sum(y_one_hot * np.log(y_pred + 1e-10)) / y_batch.size
            
            # Обратный проход
            grad = (y_pred - y_one_hot) / y_batch.size
            model.backward(grad)
            
            # Обновляем веса
            model.update()
        
        # Вычисляем точность на тестовой выборке
        val_pred = model.forward(X_test)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_accuracy = accuracy_score(y_test, val_pred_classes)
        
        # Выводим прогресс
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Сохраняем метрики
        losses.append(loss)
        accuracies.append(val_accuracy)
    
    # Визуализация результатов
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss during training')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy during training')
    # plt.show()
    # Save figure before closing
    plt.savefig("build/classification_process.png")
    plt.close()
    
    # Оцениваем модель на тестовой выборке
    y_pred = model.forward(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Выводим метрики
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\nТочность на тестовой выборке: {accuracy:.4f}")
    
    print("\nОтчет о классификации:")
    target_names = [f"Class {i}" for i in range(output_size)]
    print(classification_report(y_test, y_pred_classes, target_names=target_names))

if __name__ == "__main__":
    main()