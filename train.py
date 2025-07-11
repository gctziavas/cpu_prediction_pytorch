import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data
from model import LSTMModel
import mlflow
import mlflow.pytorch
from datetime import datetime
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("CPU_Load_Prediction")

run_name = "Run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
with mlflow.start_run(run_name=run_name):

    # Log hyperparameters - UPDATE THESE VALUES
    sequence_length = 30    # Longer memory (was 20)
    hidden_dim = 128        # More capacity (was 64)
    num_layers = 2
    lr = 0.001
    epochs = 12             # More training (was 20)
    mlflow.log_params({
        "sequence_length": sequence_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "lr": lr,
        "epochs": epochs
    })

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data("./data/mock_cpu_memory_data.csv", sequence_length)
    
    # Move to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Initialize model - UPDATE INPUT_DIM TO 4
    model = LSTMModel(input_dim=4, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Save and log inputs (X_test and y_test) before evaluation
    np.save("X_test.npy", X_test.cpu().numpy())
    np.save("y_test.npy", y_test.cpu().numpy())
    mlflow.log_artifact("X_test.npy")
    mlflow.log_artifact("y_test.npy")


    # Train loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        mlflow.log_metric("val_loss", val_loss.item(), step=epoch)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test)
        test_predictions = test_output.cpu().numpy()
    
    # Log final test metrics
    mlflow.log_metric("test_loss", test_loss.item())
    
    # Save and log the model properly
    print("💾 Saving model to MLflow...")
    
    # Save model state dict locally first
    torch.save(model.state_dict(), "model_state_dict.pth")
    mlflow.log_artifact("model_state_dict.pth")
    
    # Log the full model using MLflow PyTorch
    mlflow.pytorch.log_model(
        model, 
        "model",
        conda_env={
            'channels': ['defaults'],
            'dependencies': [
                'python=3.8',
                'pip',
                {'pip': ['torch==1.13.1', 'scikit-learn==1.1.3']}
            ],
            'name': 'mlflow-env'
        },
        registered_model_name="CPU_Load_Prediction_Model"
    )
    
    print("✅ Model saved to MLflow successfully!")

    # Optional: plot and save visualization
    plt.figure(figsize=(12, 6))
    
    # Convert predictions back to original scale for plotting
    cpu_scaler = MinMaxScaler()
    cpu_scaler.min_, cpu_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    y_test_inv = cpu_scaler.inverse_transform(y_test.cpu().numpy())
    test_pred_inv = cpu_scaler.inverse_transform(test_predictions)
    
    plt.plot(y_test_inv[:100], label="Actual CPU Usage")  # Plot first 100 samples
    plt.plot(test_pred_inv[:100], label="Predicted CPU Usage")
    plt.title("LSTM CPU Load Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("CPU Usage (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    mlflow.log_artifact("prediction_plot.png")
