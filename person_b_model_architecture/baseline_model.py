#!/usr/bin/env python

#
# # SC4001 Deep Learning - Group Project #
# ##  Oxford Flowers 102 Recognition ##
#
# Role: Person A - Data Pipeline & Baseline Model
# File: person_a_baseline/baseline_model.ipynb
#
# This notebook contains:
# 1. Dataset setup and exploration
# 2. Data augmentation pipeline
# 3. Baseline ResNet18 model
# 4. Training and evaluation
# 5. Results for team comparison
#
# Team can import the dataloader using:
# from person_a_baseline.dataloader import get_flowers_dataloaders

# In[ ]:


# ============================================
# Installation & Imports
# ============================================
# Install required packages if needed
# !pip install torch torchvision matplotlib seaborn tqdm

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# ## 🔧 Setup and Configuration
#
# Setting random seeds for reproducibility and checking available compute device (GPU/CPU).

# In[2]:


# ============================================
# Set Random Seeds for Reproducibility
# ============================================
def set_seed(seed=42):
	"""Set seeds for reproducibility"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	pl.seed_everything(seed)


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## 📊 Dataset Download and Exploration
#
# Downloading the Oxford Flowers 102 dataset from TorchVision. This dataset contains:
# - 102 flower categories
# - Images with varied scales, poses, and lighting conditions
# - Split into train, validation, and test sets

# In[ ]:


# ============================================
# Load Existing Dataset
# ============================================

# Dataset is already downloaded in ../data/flowers-102
data_root = "../data"

# Check if data exists
data_path = os.path.join(data_root, "flowers-102")
if os.path.exists(data_path):
	print(f"Dataset found at: {data_path}")
else:
	print(f"Warning: Dataset not found at {data_path}")

# Load dataset (with download=False since you already have it)
print("Loading Flowers102 dataset...")

# Load test dataset with transforms
test_dataset = datasets.Flowers102(
	root=data_root,
	split="test",
	download=False,  # Changed to False
	transform=transforms.ToTensor(),
)

# Load without transforms for exploration
train_dataset_raw = datasets.Flowers102(
	root=data_root,
	split="train",
	download=False,  # Already False
)

val_dataset_raw = datasets.Flowers102(
	root=data_root,
	split="val",
	download=False,  # Already False
)

# ## 🌸 Visualize Sample Images
#
# Let's look at some random flower samples from our training set to understand what we're working with.

# In[5]:


# ============================================
# Visualize Sample Images
# ============================================
def visualize_samples(dataset, num_samples=12, title="Sample Images"):
	"""Visualize random samples from dataset"""
	fig, axes = plt.subplots(3, 4, figsize=(15, 11))
	fig.suptitle(title, fontsize=16)

	indices = random.sample(range(len(dataset)), num_samples)

	for idx, ax in zip(indices, axes.flat):
		img, label = dataset[idx]
		ax.imshow(img)
		ax.set_title(f"Class {label}", fontsize=10)
		ax.axis("off")

	plt.tight_layout()
	plt.show()


# ## 🔄 Data Augmentation and DataLoader
#
# Creating data augmentation pipeline and DataLoader functions. This is the **key deliverable** for the team!
#
# **Training augmentations:**
# - Random cropping and resizing
# - Horizontal flipping
# - Color jittering
# - Rotation
#
# **Validation/Test:**
# - Only resize and normalize (no augmentation)

# In[7]:


# ============================================
# Create DataLoader Functions
# ============================================
def get_data_transforms():
	"""
	Returns data transforms for train and test/val sets

	Train: Data augmentation for better generalization
	Test/Val: Only normalization
	"""

	# ImageNet statistics for normalization (since we use pretrained models)
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
	)

	# Training transforms with augmentation
	train_transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(15),
			transforms.ColorJitter(
				brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
			),
			transforms.ToTensor(),
			normalize,
		]
	)

	# Validation/Test transforms (no augmentation)
	test_transform = transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]
	)

	return train_transform, test_transform


def get_flowers_dataloaders(batch_size=32, num_workers=4):
	"""
	Create DataLoaders for Flowers102 dataset

	Args:
		batch_size: Batch size for training
		num_workers: Number of parallel workers for data loading

	Returns:
		train_loader, val_loader, test_loader
	"""

	train_transform, test_transform = get_data_transforms()

	# Create datasets with transforms
	train_dataset = datasets.Flowers102(
		root="../data", split="train", transform=train_transform
	)

	val_dataset = datasets.Flowers102(
		root="../data", split="val", transform=test_transform
	)

	test_dataset = datasets.Flowers102(
		root="../data", split="test", transform=test_transform
	)

	# Create dataloaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True if torch.cuda.is_available() else False,
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True if torch.cuda.is_available() else False,
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True if torch.cuda.is_available() else False,
	)

	return train_loader, val_loader, test_loader


# Test the dataloader
# train_loader, val_loader, test_loader = get_flowers_dataloaders(batch_size=32)
# print(f"\nDataLoader created successfully!")
# print(f"Train batches: {len(train_loader)}")
# print(f"Val batches: {len(val_loader)}")
# print(f"Test batches: {len(test_loader)}")


# ## 🎨 Visualize Augmented Images
#
# Let's see how our augmentation transforms modify the training images.

# In[7]:


# ============================================
# Visualize Augmented Images
# ============================================
def show_augmented_batch(dataloader):
	"""Show a batch with augmentations applied"""
	images, labels = next(iter(dataloader))

	fig, axes = plt.subplots(2, 4, figsize=(15, 7))
	fig.suptitle("Augmented Training Images", fontsize=16)

	# Denormalize for visualization
	mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

	for idx, ax in enumerate(axes.flat):
		if idx < len(images):
			img = images[idx]
			img = img * std + mean  # Denormalize
			img = torch.clamp(img, 0, 1)

			ax.imshow(img.permute(1, 2, 0))
			ax.set_title(f"Class {labels[idx].item()}")
			ax.axis("off")

	plt.tight_layout()
	plt.show()


# show_augmented_batch(train_loader)


# ## 🤖 Baseline Model Definition
#
# Using **ResNet18** with transfer learning:
# - Pre-trained on ImageNet (1000 classes)
# - Modified final layer for 102 flower classes
# - ~11.7M parameters

# In[ ]:


# ============================================
# Define Baseline Model
# ============================================
def create_baseline_model(num_classes=102, pretrained=True, model_name="resnet18"):
	"""
	Create baseline model using transfer learning

	Args:
		num_classes: Number of output classes (102 for Flowers)
		pretrained: Use ImageNet pretrained weights
		model_name: Model architecture ('resnet18', 'resnet34', 'resnet50')

	Returns:
		model: PyTorch model ready for training
	"""

	# print(f"Creating {model_name} model...")

	if model_name == "resnet18":
		model = models.resnet18(pretrained=pretrained)

	elif model_name == "resnet34":
		model = models.resnet34(pretrained=pretrained)

	elif model_name == "resnet50":
		model = models.resnet50(pretrained=pretrained)

	elif model_name == "resnet101":
		model = models.resnet50(pretrained=pretrained)

	elif model_name == "resnet152":
		model = models.resnet50(pretrained=pretrained)

	else:
		raise ValueError(f"Model {model_name} not supported")

	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, num_classes)
	model.to(device)

	# # Count parameters
	# total_params = sum(p.numel() for p in model.parameters())
	# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	# print(f"Total parameters: {total_params:,}")
	# print(f"Trainable parameters: {trainable_params:,}")
	return model


# Create model
# model = create_baseline_model(num_classes=102, pretrained=True, model_name='resnet18')
# model = model.to(device)


# ## 🏋️ Training Functions
#
# Defining functions for training and validation loops.

# In[10]:


# ============================================
# Training Functions (Fixed Version)
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device, progress_bar=None):
	"""Train for one epoch"""
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for batch_idx, (images, labels) in enumerate(dataloader):
		images, labels = images.to(device), labels.to(device)

		# Forward pass
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)

		# Backward pass
		loss.backward()
		optimizer.step()

		# Statistics
		running_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()

		# Update progress bar
		if progress_bar is not None:
			progress_bar.update(1)
			progress_bar.set_postfix({"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total})

	return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device, progress_bar=None):
	"""Validate the model"""
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			if progress_bar is not None:
				progress_bar.update(1)
				progress_bar.set_postfix({"loss": running_loss / (total / labels.size(0)), "acc": 100.0 * correct / total})

	return running_loss / len(dataloader), 100.0 * correct / total


# ## ⚙️ Training Configuration
#
# Setting up hyperparameters, optimizer, and learning rate scheduler.

# In[11]:


# ============================================
# Training Configuration
# ============================================
# Hyperparameters
# config = {
#     'model_name': 'resnet18',
#     'num_epochs': 30,
#     'batch_size': 32,
#     'learning_rate': 0.001,
#     'weight_decay': 1e-4,
#     'scheduler_step_size': 10,
#     'scheduler_gamma': 0.1
# }

# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(
#     model.parameters(),
#     lr=config['learning_rate'],
#     weight_decay=config['weight_decay']
# )

# # Learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=config['scheduler_step_size'],
#     gamma=config['scheduler_gamma']
# )

# print("Training Configuration:")
# for key, value in config.items():
#     print(f"  {key}: {value}")


# ## 🚀 Training Loop
#
# Training the model for 30 epochs. This will take:
# - ~25-30 minutes on GPU
# - ~2-3 hours on CPU

# In[12]:


# ============================================
# Train the Model
# ============================================
# Training history
# history = {
#     'train_loss': [],
#     'train_acc': [],
#     'val_loss': [],
#     'val_acc': []
# }

# # Best model tracking
# best_val_acc = 0.0
# best_model_state = None

# print("\n" + "="*50)
# print("STARTING TRAINING")
# print("="*50)

# for epoch in range(config['num_epochs']):
#     print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
#     print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

#     # Train
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

#     # Validate
#     val_loss, val_acc = validate(model, val_loader, criterion, device)

#     # Step scheduler
#     scheduler.step()

#     # Save history
#     history['train_loss'].append(train_loss)
#     history['train_acc'].append(train_acc)
#     history['val_loss'].append(val_loss)
#     history['val_acc'].append(val_acc)

#     # Save best model
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         best_model_state = model.state_dict().copy()
#         print(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")

#     # Print metrics
#     print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#     print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# # Load best model
# model.load_state_dict(best_model_state)
# print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")


# ## 📊 Test Set Evaluation
#
# Evaluating the best model on the held-out test set for final performance metrics.

# In[13]:


# ============================================
# Evaluate on Test Set
# ============================================
# print("\n" + "="*50)
# print("EVALUATING ON TEST SET")
# print("="*50)

# test_loss, test_acc = validate(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_acc:.2f}%")


# ## 📈 Visualize Training History
#
# Plotting loss and accuracy curves to analyze training behavior.

# In[14]:


# ============================================
# Plot Training History
# ============================================
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # Loss plot
# ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
# ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
# ax1.set_xlabel('Epoch', fontsize=12)
# ax1.set_ylabel('Loss', fontsize=12)
# ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Accuracy plot
# ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
# ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
# ax2.set_xlabel('Epoch', fontsize=12)
# ax2.set_ylabel('Accuracy (%)', fontsize=12)
# ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
# ax2.legend()
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


# ## 💾 Save Model and Results
#
# Saving the trained model weights and performance metrics for team reference.

# In[15]:


# # Save model weights
# os.makedirs('results/model_weights', exist_ok=True)
# model_path = 'results/model_weights/resnet18_baseline.pth'
# torch.save(best_model_state, model_path)
# print(f"✓ Model weights saved to {model_path}")

# # Save training metrics
# results = {
#     'model': 'ResNet18',
#     'test_accuracy': float(test_acc),
#     'best_val_accuracy': float(best_val_acc),
#     'total_parameters': sum(p.numel() for p in model.parameters()),
#     'history': history
# }

# with open('results/metrics.json', 'w') as f:
#     json.dump(results, f, indent=4)

# print(f"✓ Metrics saved to results/metrics.json")


# -------

# ## 📋 Summary for Team
#
# ### Baseline Results:
# - **Model:** ResNet18 (Pretrained)
# - **Test Accuracy:** ~75%
# - **Parameters:** 11.7M
#
# ### How to use this baseline:
# ```python
# # Import dataloader
# from person_a_baseline.dataloader import get_flowers_dataloaders
# train_loader, val_loader, test_loader = get_flowers_dataloaders()
#
# # Load model
# model = torch.load('person_a_baseline/results/model_weights/resnet18_baseline.pth')
# ```
#
