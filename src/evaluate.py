import torch
from sklearn.metrics import classification_report

def evaluate_model(model, dataloader, criterion, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return acc, test_loss / len(dataloader), all_preds, all_labels, report
