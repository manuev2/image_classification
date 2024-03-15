import torch
from torch import nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
	def __init__(self):
		super(ImageClassifier, self).__init__()
		# Define the layers of the model
		self.fc1 = nn.Linear(784, 128)  # First hidden layer
		self.fc2 = nn.Linear(128, 64)   # Second hidden layer
		self.fc3 = nn.Linear(64, 10)    # Output layer

	def forward(self, x):
		# Flatten the input tensor
		x = x.view(-1, 784)  # -1 here means "infer this dimension"
		
		# Apply the layers with ReLU activations for hidden layers
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		
		# No activation for the output layer; this will be handled by the loss function
		x = self.fc3(x)
		return x