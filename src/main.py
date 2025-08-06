import os
import torch
from classifier.model import get_model
from classifier.train import train_model
from classifier.evaluate import evaluate_model
from classifier.utils import (
    get_data_loaders, show_confusion_matrix, plot_class_accuracy,
    tsne_plot, visualize_predictions
)

def save_all_visuals(model, dataloader, class_names, device):
    os.makedirs("visuals", exist_ok=True)

    print("Generating visuals...")
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    show_confusion_matrix(all_labels, all_preds, class_names)
    plot_class_accuracy(all_labels, all_preds, class_names)
    tsne_plot(model, dataloader, class_names, device)
    visualize_predictions(model, dataloader, class_names, n_per_class=5, device=device)

def main():
    # Setup
    data_dir = "dataset/split"
    batch_size = 32
    input_size = 224
    num_classes = 5
    epochs = 5
    lr = 1e-4
    model_save_path = "saved_models/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(data_dir, input_size, batch_size)

    # Build model
    model = get_model(num_classes).to(device)

    # Train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_save_path)

    # Evaluate
    acc, loss, y_pred, y_true, report = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss(), class_names, device)
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")
    print("Classification Report:")
    for cls, metrics in report.items():
        print(cls, metrics)

    # Save visuals
    save_all_visuals(model, test_loader, class_names, device)

    print("\nPipeline complete! Trained, evaluated, and saved outputs.")

if __name__ == "__main__":
    main()