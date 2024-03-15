from torchvision import datasets, transforms
import torch
import os

def load_mnist_data(batch_size=64, data_dir='./data'):
	# Define a transform to normalize the data
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	
	# Check if the dataset already exists before downloading
	train_data_path = os.path.join(data_dir, 'MNIST/raw/train-images-idx3-ubyte')
	test_data_path = os.path.join(data_dir, 'MNIST/raw/t10k-images-idx3-ubyte')
	
	if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
		download = True
	else:
		download = False

	# Download and load the training data
	trainset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

	# Download and load the test data
	testset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

	return trainloader, testloader