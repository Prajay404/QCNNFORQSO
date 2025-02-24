import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the data (assuming 'galaxy_data.csv' is available in the same directory)
@st.cache
def load_data():
    return pd.read_csv('galaxy_data.csv')

df = load_data()

# Use all columns except 'black_hole_mass' as features
features = [col for col in df.columns if col != 'black_hole_mass']
X = df[features]
y = df['black_hole_mass']

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the quantum device
n_qubits = len(features)  # One qubit for each feature
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the QCNN model
class QCNN(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layers = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = torch.tensor([quantum_circuit(x_i, self.q_layers) for x_i in x], dtype=torch.float32)
        return self.fc(x)

# Initialize the model
model = QCNN(n_qubits=n_qubits, n_layers=2)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize lists to store metrics
losses = []
accuracies = []
f1_scores = []

# Training loop
n_epochs = 80
batch_size = 32
threshold = 0.90

# Training Function (do this only once)
def train_model():
    global model, losses, accuracies, f1_scores
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate metrics
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            y_pred_binary = (y_pred > threshold).float()
            y_test_binary = (y_test > threshold).float()
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            f1 = f1_score(y_test_binary, y_pred_binary)

        losses.append(epoch_loss / (len(X_train) // batch_size))
        accuracies.append(accuracy)
        f1_scores.append(f1)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss/(len(X_train)//batch_size):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

train_model()  # Train the model before starting Streamlit app

# Streamlit UI
st.title("Black Hole Mass Prediction Using QCNN")

# Interactive input fields for varying parameters
user_inputs = []
for feature in features:
    user_input = st.slider(f"{feature}", min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()))
    user_inputs.append(user_input)

# Convert user inputs to tensor
user_inputs_tensor = torch.tensor([user_inputs], dtype=torch.float32)

# Prediction button
if st.button("Predict Mass"):
    model.eval()
    with torch.no_grad():
        y_pred = model(user_inputs_tensor)
        st.write(f"Predicted Black Hole Mass: {y_pred.item():.4f} solar masses")

# Plot the loss, accuracy, and F1 score graphs
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(131)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Accuracy and F1 Score plot
plt.subplot(132)
plt.plot(accuracies, label='Accuracy')
plt.plot(f1_scores, label='F1 Score')
plt.title('Accuracy and F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

# Confusion Matrix
plt.subplot(133)
y_pred_binary = (y_pred > threshold).float()
y_test_binary = (y_test > threshold).float()
cm = confusion_matrix(y_test_binary, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
st.pyplot()  # Display the plot
