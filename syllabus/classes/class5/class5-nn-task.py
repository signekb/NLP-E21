# Create dataset
from datasets import load_dataset
dataset = load_dataset("emotion")

print(dataset)
print(dataset[0])