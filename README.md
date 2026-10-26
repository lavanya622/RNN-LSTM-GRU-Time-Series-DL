# Recurrent Neural Networks: SimpleRNN vs LSTM vs GRU
### Deep Learning | Time-Series Forecasting | Python & TensorFlow

A practical deep learning project exploring three popular Recurrent Neural Network (RNN) architectures — **SimpleRNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)** — through individual implementations and a combined performance comparison for time-series forecasting.

This repository demonstrates how recurrent neural networks learn from sequential data, generate predictions, and compare forecasting performance using consistent evaluation metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objectives](#project-objectives)
- [Models Implemented](#models-implemented)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Running the Models](#running-the-models)
- [Methodology](#methodology)
- [Model Comparison](#model-comparison)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Key Learnings](#key-learnings)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [License](#license)

---

## Project Overview

Time-series forecasting uses historical observations to estimate future values. It is widely used in weather forecasting, sales prediction, energy consumption, traffic analysis, and other applications involving sequential data.

This project implements SimpleRNN, LSTM, and GRU models using Python and TensorFlow/Keras. Each architecture is explored individually, followed by a combined experiment that trains all three models on the same forecasting task.

The combined experiment uses a synthetic time series containing periodic patterns and random noise. Each model uses the previous 30 observations to predict the next value.

**The main goal is to understand recurrent neural network architectures through practical implementation and evaluate their performance under a shared experimental setup.**

## Project Objectives

- Understand the fundamentals of recurrent neural networks.
- Implement SimpleRNN, LSTM, and GRU as individual models.
- Prepare sequential data using sliding-window techniques.
- Apply data scaling and chronological train-test splitting.
- Train neural networks for time-series forecasting.
- Compare predictions using MAE and RMSE.
- Visualize actual values, predictions, and training history.
- Understand the strengths and limitations of different recurrent architectures.

## Models Implemented

### 1. SimpleRNN — Simple Recurrent Neural Network

SimpleRNN is a basic recurrent architecture that maintains a hidden state across time steps.

**Key characteristics:**
- Simple architecture for learning sequential patterns.
- Useful as a baseline for recurrent neural networks.
- Can struggle with long-term dependencies because of vanishing gradients.

### 2. LSTM — Long Short-Term Memory

LSTM extends recurrent networks with a cell state and gating mechanisms to help retain useful information over longer sequences.

**Key characteristics:**
- Uses input, forget, and output gates.
- Designed to learn longer-term dependencies.
- Can be useful for more complex sequential patterns.

### 3. GRU — Gated Recurrent Unit

GRU uses gating mechanisms to control how information is retained and updated, with a simpler state structure than LSTM.

**Key characteristics:**
- Uses update and reset gates.
- Often has fewer parameters than a comparable LSTM.
- Can offer a useful balance between model complexity and performance.

**Note:** No architecture is guaranteed to perform best on every dataset. Results depend on the dataset, model configuration, training process, and evaluation method.

## Repository Structure

```text
rnn-lstm-gru-time-series/
│
├── README.md
├── requirements.txt
│
├── simple_rnn_time_series.py
├── lstm_time_series.py
├── gru_time_series.py
│
└── rnn_gru_lstm_timeseries_comparison.py
```

| File | Description |
|---|---|
| `README.md` | Project documentation |
| `requirements.txt` | Required Python libraries |
| `simple_rnn_time_series.py` | Individual SimpleRNN forecasting implementation |
| `lstm_time_series.py` | Individual LSTM forecasting implementation |
| `gru_time_series.py` | Individual GRU forecasting implementation |
| `rnn_gru_lstm_timeseries_comparison.py` | Combined training and comparison of all three architectures |

*Make sure the filenames in your GitHub repository match the structure above.*

## Technologies Used

- **Python** — programming language
- **TensorFlow / Keras** — deep learning framework
- **NumPy** — numerical computation
- **Matplotlib** — data visualization
- **scikit-learn** — data preprocessing and evaluation metrics

## Installation

### Prerequisites

Install a Python version compatible with your TensorFlow release.

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/rnn-lstm-gru-time-series.git
cd rnn-lstm-gru-time-series
```

Replace `YOUR-USERNAME` with your GitHub username.

### Install Dependencies

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` file contains:

```text
tensorflow
numpy
matplotlib
scikit-learn
```

Using a virtual environment is recommended to keep project dependencies isolated.

## Running the Models

### Run SimpleRNN

```bash
python simple_rnn_time_series.py
```

### Run LSTM

```bash
python lstm_time_series.py
```

### Run GRU

```bash
python gru_time_series.py
```

### Run the Combined Comparison

```bash
python rnn_gru_lstm_timeseries_comparison.py
```

The individual scripts run their respective model implementations. The combined script trains SimpleRNN, GRU, and LSTM using the same prepared dataset, evaluates their test predictions, and generates comparison plots.

The commands assume that the corresponding Python files have been added to the repository.

## Methodology

The time-series forecasting workflow follows these steps:

1. **Data generation:** Create a synthetic periodic signal with random noise.
2. **Train-test split:** Divide the observations chronologically into training and testing periods.
3. **Data scaling:** Fit `MinMaxScaler` on training data only, then transform the test data using the same scaler.
4. **Sequence creation:** Use the previous 30 time steps to predict the next observation.
5. **Model construction:** Build a recurrent layer followed by a dense output layer.
6. **Model training:** Train with the Adam optimizer and Mean Squared Error (MSE) loss.
7. **Validation and early stopping:** Monitor validation loss to help reduce unnecessary training.
8. **Evaluation:** Calculate MAE and RMSE using the held-out test data.
9. **Visualization:** Compare actual and predicted values and inspect training and validation loss.

### Input Data Shape

The models receive three-dimensional input:

```text
(samples, time_steps, features)
```

For the combined experiment:

- `samples`: number of sequence examples
- `time_steps`: 30 previous observations
- `features`: one numerical value at each time step

## Model Comparison

The combined experiment evaluates all three models on the same time-series forecasting task.

| Model | Architecture | Main Feature |
|---|---|---|
| SimpleRNN | Basic recurrent layer | Simple sequence learning |
| LSTM | Gated recurrent layer with cell state | Designed to learn long-term dependencies |
| GRU | Gated recurrent layer | Simpler state structure than LSTM |

The combined script reports the test MAE and RMSE for each model and identifies the model with the lowest RMSE for that run.

Actual numerical results are generated when the code is executed. No model is declared the winner in advance.

## Evaluation Metrics

| Metric | Description | Preferred Result |
|---|---|---|
| MAE | Mean Absolute Error; average absolute difference between actual and predicted values | Lower |
| RMSE | Root Mean Squared Error; penalizes larger errors more strongly | Lower |
| MSE | Mean Squared Error; commonly used as the training loss | Generally lower |

MAE and RMSE are calculated after converting predictions back to the original data scale, making the errors easier to interpret.

## Visualizations

The project includes plots to help understand model behavior:

- **Time-series plot:** Displays the generated signal.
- **Actual vs predicted plot:** Compares test observations with predictions from SimpleRNN, GRU, and LSTM.
- **Training and validation loss:** Shows how the models learn over training epochs.

These visualizations help assess prediction quality and identify possible overfitting.

## Key Learnings

Through this project, you can learn how to:

- Build recurrent neural networks using TensorFlow/Keras.
- Understand the differences between SimpleRNN, LSTM, and GRU.
- Transform time-series data into supervised learning sequences.
- Scale data while avoiding test-data leakage.
- Train and evaluate deep learning models.
- Interpret MAE, RMSE, and loss curves.
- Compare model predictions under a shared experiment.

## Limitations

- The combined experiment uses synthetic data rather than a real-world benchmark.
- Results may vary depending on the environment and training run.
- A single train-test split cannot establish universal model superiority.
- The project predicts the next observation from a fixed window; it does not implement recursive multi-step forecasting.
- More rigorous evaluation may require repeated experiments, hyperparameter tuning, and walk-forward validation.

## Future Improvements

Potential extensions include:

- Apply the models to real-world weather, sales, or energy datasets.
- Add a naive forecasting baseline.
- Tune the number of recurrent units and sequence length.
- Compare model parameter counts and training times.
- Repeat experiments with multiple random seeds.
- Implement walk-forward validation.
- Save trained models and export predictions to CSV.
- Extend the task to multi-step forecasting.

## Conclusion

This project provides a practical introduction to SimpleRNN, LSTM, and GRU for time-series forecasting. It combines individual model implementations with a unified experiment to compare their predictive performance.

The repository serves as a foundation for learning recurrent neural networks and extending these techniques to real-world sequential data.

