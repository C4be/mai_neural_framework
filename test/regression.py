import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add path to the root directory
sys.path.append("/Users/cube/Documents/Development/MAI/Сошников")

# Import our classes
from framework.DataLoader import DataLoader
from framework.NeuralNetwork import NeuralNetwork, Layer, ActivationFunction, Optimizer

def normalize_feature(x):
    return (x - np.mean(x)) / np.std(x)

def main():
    # Load data
    data_loader = DataLoader("diabetes")
    
    # Shuffle and normalize data
    data_loader.shuffle()
    data_loader.map(normalize_feature)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_loader.data, data_loader.labels, test_size=0.2, random_state=42
    )
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # Reshape labels for regression
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    # Define neural network architecture
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = 1  # For regression, we predict a single continuous value
    
    # Create optimizer (Adam works well for regression too)
    optimizer = Optimizer.Adam(learning_rate=0.001)
    
    # Create neural network
    model = NeuralNetwork(optimizer)
    model.add_layer(Layer(input_size, hidden_size, ActivationFunction.ReLU))
    model.add_layer(Layer(hidden_size, hidden_size//2, ActivationFunction.ReLU))
    model.add_layer(Layer(hidden_size//2, output_size))  # No activation for regression output
    
    # Training parameters
    epochs = 200
    batch_size = 32
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    r2_scores = []
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        batches = data_loader.batch(batch_size)
        
        for X_batch, y_batch in batches:
            # Normalize batch data
            X_batch = scaler_X.transform(X_batch)
            y_batch = scaler_y.transform(y_batch.reshape(-1, 1))
            
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Calculate MSE loss
            loss = np.mean((y_pred - y_batch) ** 2)
            epoch_loss += loss
            
            # Backward pass
            grad = 2 * (y_pred - y_batch) / y_batch.shape[0]
            model.backward(grad)
            
            # Update weights
            model.update()
        
        # Calculate validation metrics
        y_pred_val = model.forward(X_test)
        val_loss = mean_squared_error(y_test, y_pred_val)
        r2 = r2_score(y_test, y_pred_val)
        
        # Store metrics
        train_losses.append(epoch_loss / len(batches))
        val_losses.append(val_loss)
        r2_scores.append(r2)
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Training Loss: {epoch_loss/len(batches):.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"R² Score: {r2:.4f}\n")
    
    # Visualize training process
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # Plot R² scores
    plt.subplot(1, 3, 2)
    plt.plot(r2_scores)
    plt.title('R² Score During Training')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    
    # Plot predictions vs actual values
    plt.subplot(1, 3, 3)
    y_pred_test = model.forward(X_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    plt.scatter(y_test_orig, y_pred_test, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 
             'r--', lw=2)
    plt.title('Predictions vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    
    # Save figure
    plt.savefig("build/regression_results.png")
    plt.close()
    
    # Final metrics
    final_mse = mean_squared_error(y_test_orig, y_pred_test)
    final_r2 = r2_score(y_test_orig, y_pred_test)
    
    print("\nFinal Results:")
    print(f"Mean Squared Error: {final_mse:.4f}")
    print(f"R² Score: {final_r2:.4f}")

if __name__ == "__main__":
    main()