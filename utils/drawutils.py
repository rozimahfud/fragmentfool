import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def draw_graph(graph,saved_name):
    mapping = dict(zip(list(graph.nodes),list(range(len(graph.nodes)))))
    graph = nx.relabel_nodes(graph, mapping)

    plt.figure(figsize=(15, 10))
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', 
            font_size=10, font_weight='bold', arrows=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/'+saved_name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def threshold_plot(y_probs,y_true):
    thresholds = np.arange(0.3, 0.7, 0.01)
    
    results = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        diff = abs(prec - rec)  # We want to MINIMIZE this value
        
        results.append((t, prec, rec, f1, acc, diff))
        print(f"Threshold: {t:.2f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, Diff: {diff:.4f}")

    # Find threshold with smallest difference
    results.sort(key=lambda x: x[5])  # Sort by difference, ascending
    best_threshold = results[0][0]
    print(f"\nBest threshold for minimum difference: {best_threshold:.2f}")
    print(f"Precision: {results[0][1]:.4f}, Recall: {results[0][2]:.4f}")
    print(f"F1-Score: {results[0][3]:.4f}, Accuracy: {results[0][4]:.4f}")
    print(f"Difference: {results[0][5]:.4f}")

    # Visualize the impact of different thresholds
    import matplotlib.pyplot as plt

    thresholds = [r[0] for r in results]
    precisions = [r[1] for r in results]
    recalls = [r[2] for r in results]
    diffs = [r[5] for r in results]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'r-', label='Recall')
    plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')

    plt.subplot(2, 1, 2)
    plt.plot(thresholds, diffs, 'k-')
    plt.axvline(x=best_threshold, color='g', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('|Precision - Recall|')
    plt.title('Difference Between Precision and Recall')

    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    plt.show()

def plot_and_save_training_metrics(train_loss, test_loss, train_acc, test_acc, filename='training_metrics.png'):
    # Convert to numpy arrays if they aren't already
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    
    # Create epochs array
    epochs = np.arange(1, len(train_loss) + 1)
    
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, test_loss, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_acc, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("figures/"+filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename}")
    plt.close() 
    # # Display the plot
    # plt.show()
    