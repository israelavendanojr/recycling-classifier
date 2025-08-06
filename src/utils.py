import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd
from collections import defaultdict
from sklearn.manifold import TSNE
from torchvision import datasets
from torch.utils.data import DataLoader
import os

def get_transforms(input_size=224):
    train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train, val_test

def get_data_loaders(data_dir, input_size=224, batch_size=32):
    train_tfms, val_test_tfms = get_transforms(input_size)

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_tfms)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_tfms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names

# Visualizations
def show_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("visuals/confusion_matrix.png")
    plt.show()


def plot_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    df = pd.DataFrame({'Class': class_names, 'Accuracy': acc})
    df.plot(x='Class', y='Accuracy', kind='bar', legend=False, ylim=(0, 1), title='Per-Class Accuracy', ylabel='Accuracy')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("visuals/class_accuracy.png")
    plt.show()


def tsne_plot(model, dataloader, class_names, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            output = model.features(images).mean([2, 3])
            features.append(output.cpu())
            labels.extend(lbls.numpy())
    features = torch.cat(features).numpy()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    label_names = [class_names[i] for i in labels]
    tsne_df = pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'Class': label_names})
    sns.scatterplot(data=tsne_df, x='x', y='y', hue='Class', palette='tab10')
    plt.title("t-SNE of Test Set Features")
    plt.tight_layout()
    plt.savefig("visuals/tsne.png")
    plt.show()


def visualize_predictions(model, dataloader, class_names, n_per_class=5, device='cpu'):
    model.eval()
    all_images, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_images.extend(images.cpu())
            all_preds.extend(preds.cpu())
            all_labels.extend(labels)
    class_to_indices = defaultdict(list)
    for i, label in enumerate(all_labels):
        class_to_indices[label.item()].append(i)
    indices = [idx for c in class_to_indices.values() for idx in random.sample(c, min(n_per_class, len(c)))]
    cols, rows = 5, (len(indices) + 4) // 5
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(indices):
        img = all_images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        pred = class_names[all_preds[idx]]
        true = class_names[all_labels[idx]]
        color = 'green' if pred == true else 'red'
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"P: {pred}\nT: {true}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("visuals/predictions_sample.png")
    plt.show()