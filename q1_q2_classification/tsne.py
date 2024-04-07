
"""
 We can also visualize how the feature representations specialize for different classes. Take 1000
random images from the test set of PASCAL, and extract ImageNet (finetuned) features from
those images. Compute a 2D t-SNE (use sklearn) projection of the features, and plot them with
each feature color-coded by the GT class of the corresponding image. If multiple objects are
active in that image, compute the color as the “mean” color of the different classes active in that
image. Add a legend explaining the mapping from color to object class.
"""

import torch
# import trainer
import utils
# from simple_cnn import SimpleCNN
from train_q2 import ResNet
from voc_dataset import VOCDataset
import numpy as np
# import random
# import sklearn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
# import torchvision
# import torch.nn as nn
# import os

import warnings
warnings.filterwarnings("ignore")

# Import the saved model located at the path: 'q1_q2_classification/checkpoint-model-epoch2.pth'
# Load the model
model = torch.load('checkpoint-model-epoch4.pth')
# odel.resnet.fc = torch.nn.Identity()

model = torch.nn.Sequential(*list(model.children())[:-1],
                           torch.nn.Flatten())

for param in model.parameters():
    param.requires_grad = False  # freeze the model

# Ensure model is in correct mode and on right device
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # move the model to the device

# Load the test dataset of 1000 random images
test_loader = utils.get_data_loader(name='voc', train=False, batch_size=1000, split='test', inp_size=224)

# Extract ImageNet (finetuned) features from those images
# Feauture represent the output of the model before the final classification layer
# Labels represent the ground truth labels
features = []
labels = []

idx = 0
print('Extracting features...')
for batch_idx, (data, target, wgt) in enumerate(test_loader):  # iterate over the test data
    data = data.to(device)
    output = model(data)
    features.append(output.detach().cpu().numpy())
    labels.append(target.detach().cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

print('Computing t-SNE...')
# Compute a 2D t-SNE (use sklearn) projection of the features
tsne = TSNE(n_components=2, random_state=0)  # random_state=0 for reproducibility
features_2d = tsne.fit_transform(features)  # returns a 2D array of the features

# Generating colors and computing the mean color of the different classes active in that image
num_classes = len(VOCDataset.CLASS_NAMES)
color_map = cm.get_cmap('tab20', num_classes)
class_colors = color_map(np.arange(num_classes))
class_colors = class_colors[:, :3]  # remove alpha channel

# Compute the mean color of the different classes active in that image
mean_colors = np.zeros((labels.shape[0], 3))
for i in range(labels.shape[0]):
    active_classes = np.where(labels[i] == 1)[0]
    if len(active_classes) > 0:
        mean_colors[i] = np.mean(class_colors[active_classes], axis=0)
        mean_colors[i] = mean_colors[i] / np.linalg.norm(mean_colors[i])  # normalize
    else:
        mean_colors[i] = np.array([0, 0, 0])

# Plot them with each feature color-coded by the GT class of the corresponding image
print('Plotting...')
plt.figure(figsize=(10, 8))
for point, color in zip(features_2d, mean_colors):
    plt.scatter(point[0], point[1], color=color, alpha=0.7)

# Create a legend
for i, class_name in enumerate(VOCDataset.CLASS_NAMES):
    plt.scatter([], [], color=class_colors[i], label=class_name)

plt.legend(frameon=True, title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("2D t-SNE Visualization of Image Features (Last Conv Layer)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.show()
# plt.savefig('2D_t_SNE.png')
