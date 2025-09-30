import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.dataset import PlantDataset

def evaluate():
    # Load model
    model = tf.keras.models.load_model('models/best_model.keras')
    
    # Load test data
    dataset = PlantDataset('data')
    test_gen = dataset.load_test_data()
    
    # Evaluate
    results = model.evaluate(test_gen)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # Predictions
    y_true = test_gen.classes
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=dataset.class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=dataset.class_names,
               yticklabels=dataset.class_names)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    evaluate()