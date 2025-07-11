# LSTM-based CPU Usage Prediction

This project demonstrates a complete workflow for predicting CPU usage using LSTM neural networks with PyTorch. The project includes data preprocessing, model training with MLflow tracking, evaluation, and future forecasting capabilities.

## Project Structure

```
cpu_prediction_pytorch/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── model.py                           # LSTM model definition
├── data_loader.py                     # Data loading and preprocessing utilities
├── train.py                          # Standalone training script
├── lstm_training_prediction.ipynb    # Interactive Jupyter notebook
├── data/
│   └── mock_cpu_memory_data.csv      # Sample dataset
└── trained_models/                   # Directory for saved models (created during training)
```

## Features

- **LSTM Neural Network**: Multi-layer LSTM architecture for time series prediction
- **Feature Engineering**: Incorporates temporal features (hour of day, day of week)
- **MLflow Integration**: Experiment tracking and model versioning
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Future Forecasting**: Multi-step ahead prediction capability
- **Interactive Notebook**: Complete workflow demonstration

## Setup Instructions

### 1. Clone or Navigate to the Project

```bash
cd /home/ubuntu/cpu_prediction_pytorch
```

### 2. Create a Virtual Environment

#### Option A: Using `venv` (Recommended)

```bash
# Create virtual environment
python3 -m venv lstm_env

# Activate virtual environment
source lstm_env/bin/activate

# Verify activation (you should see (lstm_env) in your prompt)
which python
```

#### Option B: Using `conda`

```bash
# Create conda environment
conda create -n lstm_env python=3.8 -y

# Activate environment
conda activate lstm_env
```

### 3. Upgrade pip (Recommended)

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test other key packages
python -c "import mlflow, sklearn, pandas, matplotlib; print('All packages installed successfully!')"
```

## Requirements

The project requires the following Python packages (see `requirements.txt` for specific versions):

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Plotting and visualization
- **MLflow**: Experiment tracking and model management
- **Jupyter**: Interactive notebook environment

### MLflow Installation and Setup

For detailed MLflow installation instructions and advanced configuration options, refer to the official documentation:
- **MLflow Installation Guide**: https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
- **MLflow Tracking Documentation**: https://mlflow.org/docs/latest/tracking.html

The basic installation is included in our `requirements.txt`, but for production deployments or advanced features, consult the official documentation.

## Usage

### Option 1: Interactive Jupyter Notebook (Recommended)

1. **Start MLflow tracking server** (in a separate terminal):
   ```bash
   # Activate your virtual environment first
   source lstm_env/bin/activate
   
   # Start MLflow server
   mlflow server --host 127.0.0.1 --port 8080
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   # In the project directory with activated environment
   jupyter notebook lstm_training_prediction.ipynb
   ```

3. **Run the notebook cells** step by step to:
   - Load and visualize the data
   - Train the LSTM model
   - Evaluate performance
   - Make predictions
   - Visualize results

### Option 2: Run Training Script

```bash
# Ensure MLflow server is running (see step 1 above)
python train.py
```

## Model Configuration

Key hyperparameters that can be adjusted:

- `sequence_length`: Length of input sequences (default: 30)
- `hidden_dim`: LSTM hidden layer size (default: 128)
- `num_layers`: Number of LSTM layers (default: 2)
- `learning_rate`: Training learning rate (default: 0.001)
- `num_epochs`: Training epochs (default: 20)

## Data Format

The model expects CSV data with the following columns:
- `timestamp`: Date and time
- `cpu_usage`: CPU usage percentage (0-100)
- `memory_usage`: Memory usage percentage (0-100)

Additional temporal features (hour of day, day of week) are automatically generated.

## MLflow Tracking

The project uses MLflow for experiment tracking:

1. **Start MLflow UI**:
   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

2. **Access MLflow UI**: Open http://127.0.0.1:8080 in your browser

3. **View experiments**: Track metrics, parameters, and model artifacts

## Outputs

After training, the following artifacts are generated:

- **Model files**: `trained_lstm_model.pth`
- **MLflow artifacts**: Logged in `mlruns/` directory
- **Visualizations**: Training curves, prediction plots
- **Test data**: `X_test.npy`, `y_test.npy` for reproducibility

## Troubleshooting

### Common Issues

1. **MLflow connection error**:
   ```bash
   # Ensure MLflow server is running
   mlflow server --host 127.0.0.1 --port 8080
   ```

2. **CUDA not available**:
   - The code automatically falls back to CPU
   - For GPU support, install PyTorch with CUDA

3. **Module not found errors**:
   ```bash
   # Ensure virtual environment is activated
   source lstm_env/bin/activate
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

4. **Jupyter notebook issues**:
   ```bash
   # Install Jupyter in the virtual environment
   pip install jupyter
   
   # Install kernel
   python -m ipykernel install --user --name=lstm_env
   ```

### Deactivating Virtual Environment

When you're done working:

```bash
# For venv
deactivate

# For conda
conda deactivate
```

## Model Performance

Expected performance metrics on the sample dataset:
- **RMSE**: ~2-5% CPU usage
- **MAE**: ~1-3% CPU usage  
- **R² Score**: >0.8

## Contributing

To contribute to this project:

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

This project is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for detailed examples
3. Check MLflow logs for training issues
