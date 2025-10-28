# Energy Efficiency Prediction using PyTorch

## Overview
This project focuses on predicting the **energy efficiency of buildings** based on architectural and environmental parameters using a deep learning model built in PyTorch.  
The goal is to model and minimize the energy load (heating and cooling requirements) through accurate prediction.

## Objective
Energy efficiency is one of the most crucial aspects of sustainable architecture and smart energy systems.  
By leveraging machine learning, this project aims to estimate energy consumption patterns and provide insights for energy optimization.

## Dataset
- **Source:** UCI Machine Learning Repository — *Energy Efficiency Dataset (ENB2012)*
- **Features:** Building attributes such as relative compactness, surface area, wall area, roof area, height, glazing area, and orientation.
- **Targets:** Heating load (Y1) and Cooling load (Y2)

## Model Architecture
- **Framework:** PyTorch  
- **Model Type:** Feedforward Neural Network (Fully Connected Layers)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam  
- **Training Epochs:** 100  

The model learns to predict the heating and cooling load values based on input parameters.  
Loss decreases steadily with each epoch, indicating proper convergence.

## Training Summary
The training process demonstrates smooth loss reduction from approximately **6940 → 132 MSE** across 100 epochs.

| Metric | Initial Value | Final Value |
|---------|----------------|-------------|
| Epoch | 1 | 100 |
| Loss (MSE) | 6940.67 | 132.36 |

## Technologies Used
- **Python 3.10+**
- **Pandas** – data preprocessing  
- **NumPy** – numerical computation  
- **Matplotlib** – data visualization  
- **PyTorch** – deep learning framework  

## How to Run
```bash
# Clone the repository
git clone https://github.com/NikhilRaman12/Energy-_efficiency_model_pytorch.git

# Navigate into the project directory
cd Energy-_efficiency_model_pytorch

# Install dependencies
pip install -r requirements.txt

# Run the training script
python energy_model.py
