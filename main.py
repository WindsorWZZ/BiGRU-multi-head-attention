import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import warnings
import locale
from torchviz import make_dot
import torch.nn.functional as F

# Set locale to UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        locale.setlocale(locale.LC_ALL, 'C')

# Import our custom modules
from model import BiGRUAttentionModel
from polygon_area_metric import polygon_area_metric

# Set seeds for reproducibility
def set_seed(seed=9365):
    if seed is None:
        # Generate a random seed between 1 and 10000
        seed = random.randint(1, 10000)
        
    # Print the seed for future reproducibility
    print(f"\nUsing random seed: {seed}")
    print(f"To reproduce experimental results, use the same seed value.")
    
    # Set seeds for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed

# Define a function to plot training progress
def plot_training_progress(train_losses_min, train_losses_max, train_accuracies_min, train_accuracies_max):
    # # Set Chinese font support
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    # plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate means for plotting
    train_accuracies_mean = [(min_val + max_val) / 2 for min_val, max_val in zip(train_accuracies_min, train_accuracies_max)]
    train_losses_mean = [(min_val + max_val) / 2 for min_val, max_val in zip(train_losses_min, train_losses_max)]
    
    # Plot accuracy with fill between min and max values
    iterations = range(1, len(train_accuracies_min) + 1)
    ax1.plot(iterations, train_accuracies_mean, 'r-', linewidth=2, label='Average Accuracy')
    # ax1.plot(iterations, train_accuracies_max, 'r-', linewidth=1, alpha=0.6)
    # ax1.plot(iterations, train_accuracies_min, 'r-', linewidth=1, alpha=0.6)
    # ax1.fill_between(iterations, train_accuracies_min, train_accuracies_max, color='red', alpha=0.3)
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training Set Accuracy Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(1, len(train_accuracies_min))
    min_acc = min(train_accuracies_min) * 0.95
    max_acc = max(train_accuracies_max) * 1.05
    ax1.set_ylim(min_acc, max_acc)
    
    # Plot loss with fill between min and max values
    ax2.plot(iterations, train_losses_mean, 'b-', linewidth=2, label='Average Loss')
    # ax2.plot(iterations, train_losses_max, 'b-', linewidth=1, alpha=0.6)
    # ax2.plot(iterations, train_losses_min, 'b-', linewidth=1, alpha=0.6)
    # ax2.fill_between(iterations, train_losses_min, train_losses_max, color='blue', alpha=0.3)
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Set Loss Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(1, len(train_losses_min))
    # Set y-axis range to make the fill area more visible
    min_loss = min(train_losses_min) * 0.95
    max_loss = max(train_losses_max) * 1.05
    ax2.set_ylim(min_loss, max_loss)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot prediction results
def plot_prediction_results(y_true, y_pred, title, accuracy):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(y_true) + 1), y_true, 'r-*', label='True Values')
    plt.plot(range(1, len(y_pred) + 1), y_pred, 'b-o', label='BiGRU-Multihead-Attention Predictions')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Classification Results')
    plt.title(f'{title}\nAccuracy = {accuracy:.2f}%')
    plt.grid(True)
    plt.savefig(f'prediction_results_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Samples')
    
    # Set axis labels
    classes = np.unique(np.concatenate((y_true, y_pred)))
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot ROC curve for multi-class
def plot_roc_curve_multi(y_true, y_pred, n_classes):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calculate micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(8, 8))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='red', linestyle='--', lw=2)
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k-', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save the figure
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Add a simple model visualization function
def visualize_model(model, input_shape, filename='model_visualization.png'):
    """
    Use torchviz to generate a simple model structure visualization and save it to a file
    
    Parameters:
    model: PyTorch model
    input_shape: Input shape, e.g., (batch_size, input_dim)
    filename: Name of the file to save
    """
        
    # Create a sample input
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    
    # Get model output
    y = model(x)
    
    # Create computational graph visualization
    dot = make_dot(y, params=dict(list(model.named_parameters())))
    
    # Set graph format and size
    dot.attr('graph', rankdir='TB', size='12,12')  # Top-to-bottom layout, set size
    dot.attr('node', shape='box', style='filled', color='lightblue')
    
    # Render and save the image
    dot.render(filename.replace('.png', ''), format='png', cleanup=True)
    print(f"Model visualization saved to: {filename}")
    return True

def main():
    # Turn off warnings
    warnings.filterwarnings('ignore')
    
    # Clear plots, etc.
    plt.close('all')
    
    # Set seed for reproducibility with a random value
    random_seed = set_seed()
    
    # CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    try:
        print("Attempting to load data.xlsx...")
        data = pd.read_excel('data.xlsx', header=0).values
        print(f"Successfully loaded data with shape: {data.shape}")
        
        if data.size == 0:
            raise ValueError("Loaded data is empty")
            
        if data.shape[1] < 2:
            raise ValueError(f"Data must have at least 2 columns (features + labels). Found {data.shape[1]} columns")
            
        # Create label mapping dictionary, ensuring order: Normalcy -> 0, Warning -> 1, Malfunction -> 2
        label_order = ['Normalcy', 'Warning', 'Malfunction']
        label_map = {label: idx for idx, label in enumerate(label_order)}
        # Create reverse mapping dictionary
        reverse_label_map = {idx: label for label, idx in label_map.items()}
        print("\nLabel mapping:")
        for label, idx in label_map.items():
            print(f"{label} -> {idx}")
            
        # Convert labels to numbers
        data[:, -1] = np.array([label_map[label] for label in data[:, -1]])
            
    except FileNotFoundError:
        print("Error: data.xlsx not found in the current directory")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir())
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Analyze data
    num_class = len(np.unique(data[:, -1]))  # Number of classes (last column has the labels)
    num_dim = data.shape[1] - 1  # Feature dimensions
    num_res = data.shape[0]  # Number of samples
    
    print(f"\nData analysis:")
    print(f"- Number of samples: {num_res}")
    print(f"- Number of features: {num_dim}")
    print(f"- Number of classes: {num_class}")
    print(f"- Unique labels: {np.unique(data[:, -1])}")
    
    train_ratio = 0.7  # Train ratio
    flag_confusion = True  # Whether to show confusion matrix
    
    # Shuffle data
    np.random.shuffle(data)
    
    # Initialize variables to store data
    X_train, X_test = [], []
    y_train, y_test = [], []
    
    # Split data by class to ensure balance
    unique_labels = np.unique(data[:, -1])
    for label in unique_labels:
        class_data = data[data[:, -1] == label, :]
        class_size = class_data.shape[0]
        train_size = int(train_ratio * class_size)
        
        print(f"Class {label}: {class_size} samples (train: {train_size}, test: {class_size - train_size})")
        
        X_train.append(class_data[:train_size, :-1])
        y_train.append(class_data[:train_size, -1])
        
        X_test.append(class_data[train_size:, :-1])
        y_test.append(class_data[train_size:, -1])
    
    # Combine all classes
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    print(f"\nFinal data split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Get sample sizes
    M = X_train.shape[0]  # Train samples
    N = X_test.shape[0]   # Test samples
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train.astype(int)).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test.astype(int)).to(device)
    
    # Reshape input for the model - as per MATLAB reshaping to [num_dim, 1, 1, M]
    # But for PyTorch we'll use [batch_size, num_dim]
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = BiGRUAttentionModel(
        input_dim=num_dim,
        hidden_dim=5,
        num_classes=num_class,
        num_heads=2
    ).to(device)
    
    # Print model summary
    print(model)
    
    # Add model visualization - called after model initialization
    print("\nGenerating model visualization...")
    visualize_model(model, input_shape=(1, num_dim), filename='model_visualization.png')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,  # Initial learning rate
        weight_decay=0.001  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=400,  # LearnRateDropPeriod - adjusted for better performance
        gamma=0.2  # LearnRateDropFactor - adjusted for better performance
    )
    
    # Training loop
    num_epochs = 500
    train_losses_min = []
    train_losses_max = []
    train_accuracies_min = []
    train_accuracies_max = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # For recording batch statistics within each epoch
        batch_losses = []
        batch_accuracies = []
        
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (GradientThreshold = 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate metrics for this batch
            batch_loss = loss.item()  # Single batch loss (not multiplied by batch size)
            batch_losses.append(batch_loss)
            
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == targets).sum().item()
            batch_total = targets.size(0)
            batch_accuracy = 100 * batch_correct / batch_total
            batch_accuracies.append(batch_accuracy)
            
            # Accumulate overall statistics
            epoch_loss += batch_loss * inputs.size(0)  # Accumulated loss needs to consider batch size
            total += batch_total
            correct += batch_correct
        
        # Update learning rate
        scheduler.step()
        
        # If this epoch has multiple batches, record max and min values
        if len(batch_losses) > 0:
            train_losses_min.append(min(batch_losses))
            train_losses_max.append(max(batch_losses))
            train_accuracies_min.append(min(batch_accuracies))
            train_accuracies_max.append(max(batch_accuracies))
        else:
            # If there's only one batch (unlikely), add some variation for visualization
            avg_loss = epoch_loss / len(train_dataset)
            avg_accuracy = 100 * correct / total
            train_losses_min.append(avg_loss * 0.95)
            train_losses_max.append(avg_loss * 1.05)
            train_accuracies_min.append(avg_accuracy * 0.95)
            train_accuracies_max.append(avg_accuracy * 1.05)
        
        # Calculate average loss and accuracy (only for log output)
        avg_epoch_loss = epoch_loss / len(train_dataset)
        avg_accuracy = 100 * correct / total
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')
    
    # Plot training progress
    plot_training_progress(train_losses_min, train_losses_max, train_accuracies_min, train_accuracies_max)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        # Training set
        outputs_train = model(X_train_tensor)
        _, predicted_train = torch.max(outputs_train, 1)
        accuracy_train = 100 * (predicted_train == y_train_tensor).sum().item() / len(y_train_tensor)
        
        # Test set
        outputs_test = model(X_test_tensor)
        _, predicted_test = torch.max(outputs_test, 1)
        accuracy_test = 100 * (predicted_test == y_test_tensor).sum().item() / len(y_test_tensor)
    
    # Convert PyTorch tensors to numpy for further analysis
    y_pred_train = predicted_train.cpu().numpy()
    y_true_train = y_train_tensor.cpu().numpy()
    
    y_pred_test = predicted_test.cpu().numpy()
    y_true_test = y_test_tensor.cpu().numpy()
    
    # Plot prediction results
    def convert_labels_to_original(labels):
        return np.array([reverse_label_map[label] for label in labels])
    
    # Plot confusion matrices
    plot_confusion_matrix(y_true_train, y_pred_train, 'Confusion Matrix for Training Data')
    plot_confusion_matrix(y_true_test, y_pred_test, 'Confusion Matrix for Test Data')
    
    # Plot prediction results with original labels
    plot_prediction_results(
        convert_labels_to_original(y_true_train), 
        convert_labels_to_original(y_pred_train), 
        'Training Set Prediction Results', 
        accuracy_train
    )
    plot_prediction_results(
        convert_labels_to_original(y_true_test), 
        convert_labels_to_original(y_pred_test), 
        'Test Set Prediction Results', 
        accuracy_test
    )
    
    # Calculate polygon metrics
    metrics_test = polygon_area_metric(y_true_test, y_pred_test, is_plot=True)
    
    # Plot ROC curves for multi-class
    plot_roc_curve_multi(y_true_test, y_pred_test, num_class)
    
    print(f"\nTraining Set Accuracy: {accuracy_train:.2f}%")
    print(f"Test Set Accuracy: {accuracy_test:.2f}%")

if __name__ == '__main__':
    main() 