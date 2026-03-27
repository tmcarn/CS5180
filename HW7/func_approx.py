import numpy as np
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

X = np.linspace(-10, 10, 500)

y = 1 + X ** 2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

X_tensor = torch.from_numpy(X).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        self.seq_modules = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3  
            )

    def forward(self, x):
        out = self.seq_modules(x)
        return out
    
def train(model, dataloader, loss_fn, optimizer, epochs):
    loss_history = []
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            preds = model(batch_X.unsqueeze(1))
            l = loss_fn(preds.squeeze(), batch_y)
            loss_history.append(l.item())
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {l.item():.4f}")

    return loss_history

def predict(model, X):
    with torch.no_grad():
        x_tensor = torch.from_numpy(X).float().to(device)
        preds = model(x_tensor.unsqueeze(1))
        return preds.squeeze().cpu().numpy()
    
def plot_loss(loss_history, labels):
    plt.figure(figsize=(10, 6))
    
    for i, (loss, label) in enumerate(zip(loss_history, labels)):
        plt.plot(loss, label=label)
    
    plt.title("Batch Loss Over Epochs for Different Hidden Layer Sizes")
    plt.xlabel("Iterations")
    plt.ylabel("Batch Loss")
    plt.legend()
    plt.grid()
    plt.show()

def plot_function(X, y, preds, labels):
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label="True Function", color="blue")

    for pred, label in zip(preds, labels):
        plt.plot(X, pred, label=label, linestyle="--")

    plt.axvline(x=-10, color='r')
    plt.axvline(x=10, color='r')

    plt.title("Function Approximation with MLP")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

hidden_widths = [8, 16, 64, 128]
learning_rate = 1e-2
batch_size = 32
epochs = 50

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

models = []  # To store models for different hidden widths
losses = []  # To store loss histories for different hidden widths
for hw in hidden_widths:
    model = MLP(input_size=1, hidden_size=hw, output_size=1).to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = train(model, dataloader, loss, optimizer, epochs)
    models.append(model)
    losses.append(loss_history)


plot_loss(losses, [f"Hidden Layer Size: {hw}" for hw in hidden_widths])

model.eval()

preds = []
for model in models:
    pred = predict(model, X)
    preds.append(pred)

plot_function(X, y, preds, [f"Hidden Layer Size: {hw}" for hw in hidden_widths])


X_large = np.linspace(-30, 30, 500)

y_large = 1 + X_large ** 2

best_model = models[-1]  # Assuming the model with the largest hidden layer size performs best
best_model.eval()
pred_large = predict(best_model, X_large)

plot_function(X_large, y_large, [pred_large], ["Best Model Prediction"])


