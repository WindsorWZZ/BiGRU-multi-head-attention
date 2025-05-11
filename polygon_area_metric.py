import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    cohen_kappa_score
)

def polygon_area_metric(actual_label, predicted_label, is_plot=True):
    """
    Python implementation of the MATLAB polygonareametric function.
    This evaluates classification performance with a single metric called polygon area metric (PAM).
    
    Parameters:
    -----------
    actual_label : array-like
        True labels of the samples
    predicted_label : array-like
        Predicted labels of the samples
    is_plot : bool, optional
        Whether to save the resultant figure, default is True
        
    Returns:
    --------
    metrics : dict
        Dictionary containing various performance metrics:
        - PA: Polygon Area
        - CA: Classification Accuracy
        - SE: Sensitivity
        - SP: Specificity
        - AUC: Area Under Curve
        - K: Kappa
        - F_M: F-measure
    """
    # Ensure inputs are numpy arrays
    actual_label = np.array(actual_label)
    predicted_label = np.array(predicted_label)
    
    # Calculate confusion matrix
    cm = confusion_matrix(actual_label, predicted_label)
    
    # Calculate metrics
    # Classification Accuracy
    CA = accuracy_score(actual_label, predicted_label)
    
    # For multi-class, calculate average sensitivity and specificity
    SE = 0
    SP = 0
    n_classes = len(np.unique(actual_label))
    
    for i in range(n_classes):
        # True positives for current class
        TP = cm[i, i]
        # False negatives for current class
        FN = np.sum(cm[i, :]) - TP
        # False positives for current class
        FP = np.sum(cm[:, i]) - TP
        # True negatives for current class
        TN = np.sum(cm) - (TP + FP + FN)
        
        # Calculate sensitivity and specificity for current class
        class_SE = TP / (TP + FN) if (TP + FN) > 0 else 0
        class_SP = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        SE += class_SE
        SP += class_SP
    
    # Average sensitivity and specificity
    SE = SE / n_classes
    SP = SP / n_classes
    
    # Precision and Recall (using macro average)
    Pr = precision_score(actual_label, predicted_label, average='macro')
    Re = recall_score(actual_label, predicted_label, average='macro')
    
    # F-measure
    F_M = f1_score(actual_label, predicted_label, average='macro')
    
    # Kappa coefficient
    K = cohen_kappa_score(actual_label, predicted_label)
    
    # AUC (using one-vs-rest approach)
    AUC_value = 0
    for i in range(n_classes):
        # Convert to binary classification for current class
        y_true_binary = (actual_label == i).astype(int)
        y_pred_binary = (predicted_label == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        AUC_value += auc(fpr, tpr)
    AUC_value = AUC_value / n_classes
    
    # Define the vertices of the polygon (widest polygon)
    A1, A2, A3, A4, A5, A6 = 1, 1, 1, 1, 1, 1
    
    a = np.array([-A1, -A2/2, A3/2, A4, A5/2, -A6/2, -A1])
    b = np.array([0, -(A2*np.sqrt(3))/2, -(A3*np.sqrt(3))/2, 0, (A5*np.sqrt(3))/2, (A6*np.sqrt(3))/2, 0])
    
    # Plot the widest polygon if requested
    if is_plot:
        plt.figure(figsize=(10, 10))
        plt.plot(a, b, '--bo', linewidth=1.3)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.gca().set_aspect('equal')
    
    # Calculate the vertices of the actual polygon based on metrics
    x = np.array([-CA, -SE/2, SP/2, AUC_value, K/2, -F_M/2, -CA])
    y = np.array([0, -(SE*np.sqrt(3))/2, -(SP*np.sqrt(3))/2, 0, (K*np.sqrt(3))/2, (F_M*np.sqrt(3))/2, 0])
    
    # Plot the polygon with metrics if requested
    if is_plot:
        plt.plot(x, y, '-ko', linewidth=1)
        plt.fill(x, y, color=(0.8706, 0.9216, 0.9804), alpha=0.7)
        
        # Add labels and other elements to the plot
        plt.plot(0, 0, 'r+')
        
        plt.plot([0, -A1], [0, 0], '--ko')
        plt.text(-A1-0.3, 0, 'CA', fontweight='bold')
        
        plt.plot([0, -A2/2], [0, -(A2*np.sqrt(3))/2], '--ko')
        plt.text(-0.59, -1.05, 'SE', fontweight='bold')
        
        plt.plot([0, A3/2], [0, -(A3*np.sqrt(3))/2], '--ko')
        plt.text(0.5, -1.05, 'SP', fontweight='bold')
        
        plt.plot([0, A4], [0, 0], '--ko')
        plt.text(A4+0.08, 0, 'AUC', fontweight='bold')
        
        plt.plot([0, A5/2], [0, (A5*np.sqrt(3))/2], '--ko')
        plt.text(0.5, 1.05, 'K', fontweight='bold')
        
        plt.plot([0, -A6/2], [0, (A6*np.sqrt(3))/2], '--ko')
        plt.text(-0.65, 1.05, 'FM', fontweight='bold')
        
        plt.grid(True)
        plt.title('Polygon Area Metric (PAM) Visualization')
        
        # Save the figure
        plt.savefig('pam_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate polygon area
    n = len(x)
    p_area = 0
    for i in range(n-1):
        p_area += (x[i] + x[i+1]) * (y[i] - y[i+1])
    p_area = abs(p_area) / 2
    
    # Normalize the polygon area (2.59807 is the area of the widest polygon)
    PA = p_area / 2.59807
    
    # Store all metrics in a dictionary
    metrics = {
        'PA': PA,
        'CA': CA,
        'SE': SE,
        'SP': SP,
        'AUC': AUC_value,
        'K': K,
        'F_M': F_M
    }
    
    # Print the metrics
    categories = ['Polygon Area Metric', 'Classification Accuracy', 'Sensitivity', 'Specificity', 
                 'Area Under Curve', 'Kappa Coefficient', 'F-measure']
    values = [PA, CA, SE, SP, AUC_value, K, F_M]
    
    print('Performance Metrics:')
    for cat, val in zip(categories, values):
        print(f'{cat:>23}: {val:.2f}')
    
    return metrics 