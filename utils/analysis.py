import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def analyze_classification_results(results_list: list, f_score_beta: float = 1.0, display: bool = True) -> dict:
    labels = ['no_graffiti', 'with_graffiti']
    y_true = ['with_graffiti' if results['gt'] else 'no_graffiti' for results in results_list]
    y_pred = ['with_graffiti' if results['classification'] else 'no_graffiti' for results in results_list]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        plt.show()

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn)    # false classifications of graffiti out of all non-graffiti images
    true_negative_rate = tn / (fp + tn)     # true classifications of non-graffiti out of all non-graffiti images
    false_negative_rate = fn / (tp + fn)    # false classifications of non-graffiti out of all graffiti images
    true_positive_rate = tp / (tp + fn)     # true classification of graffiti out of all graffiti images

    precision = tp / (tp + fp)              # out of all positive classifications how many were correct
    recall = tp / (tp + fn)                 # out of all positives how many were classified as such
    f_score = (1 + f_score_beta**2) * precision * recall / (f_score_beta**2 * precision + recall)

    accuracy = (tp + tn) / (tp + tn + fp + fn)  # how many observations were correctly classified

    metrics = {
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp,
        "False Positive Rate": round(false_positive_rate, 4),
        "True Negative Rate": round(true_negative_rate, 4),
        "False Negative Rate": round(false_negative_rate, 4),
        "True Positive Rate": round(true_positive_rate, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        f"F_{f_score_beta} Score": round(f_score, 4),
        "Accuracy": round(accuracy, 4),
    }

    return metrics
