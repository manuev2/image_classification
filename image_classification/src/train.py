import torch
import torch.nn as nn
import torch.optim as optim
from image_classification.src.load_data import load_mnist_data
from image_classification.src.model import ImageClassifier

trainloader, testloader = load_mnist_data(batch_size=64)

model = ImageClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 10
for epoch in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		# Zero the gradients
		optimizer.zero_grad()
		
		# Forward pass
		outputs = model(images)
		
		# Calculate loss
		loss = criterion(outputs, labels)
		
		# Backward pass and optimize
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
	
	print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(trainloader)}")

correct = 0
total = 0
with torch.no_grad():
	for images, labels in testloader:
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
torch.save(model.state_dict(), 'models/image_classifier_state_dict.pth')
