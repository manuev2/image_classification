import unittest
import torch
from image_classification.src.model import ImageClassifier
from image_classification.src.load_data import load_mnist_data

class TestImageClassifier(unittest.TestCase):
	def setUp(self):
		self.trainloader, self.testloader = load_mnist_data(batch_size=64)
		self.model = ImageClassifier()

	def test_forward_pass(self):
		# Check if the model can perform a forward pass
		images, labels = next(iter(self.trainloader))
		outputs = self.model(images)
		# Check if output size matches the batch size and number of classes
		self.assertEqual(outputs.size(), (64, 10))

if __name__ == '__main__':
	unittest.main()