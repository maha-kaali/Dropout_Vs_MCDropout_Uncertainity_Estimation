"""
Helper script to load and visualize saved training results
"""

import pickle
import matplotlib.pyplot as plt
import os

def load_metrics(metrics_path):
    """Load pickled metrics"""
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def plot_all_metrics(metrics, save_path=None):
    """Plot comprehensive training metrics"""
    epochs = range(1, metrics['epochs'] + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Test Accuracy Comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics['acc_no'], 'b-o', label='No Dropout', linewidth=2, markersize=4)
    ax1.plot(epochs, metrics['acc_do_det'], 'r-s', label='Dropout (Deterministic)', linewidth=2, markersize=4)
    ax1.plot(epochs, metrics['acc_do_mc'], 'g-^', label='Dropout (MC)', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss Comparison
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['train_loss_no'], 'b-o', label='No Dropout', linewidth=2, markersize=4)
    ax2.plot(epochs, metrics['train_loss_do'], 'r-s', label='With Dropout', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test Loss Comparison
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics['test_loss_no'], 'b-o', label='No Dropout', linewidth=2, markersize=4)
    ax3.plot(epochs, metrics['test_loss_do'], 'r-s', label='With Dropout', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Test Loss', fontsize=12)
    ax3.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Train vs Test Loss (Overfitting Analysis)
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics['train_loss_no'], 'b--', label='Train (No Dropout)', linewidth=2, alpha=0.7)
    ax4.plot(epochs, metrics['test_loss_no'], 'b-', label='Test (No Dropout)', linewidth=2)
    ax4.plot(epochs, metrics['train_loss_do'], 'r--', label='Train (Dropout)', linewidth=2, alpha=0.7)
    ax4.plot(epochs, metrics['test_loss_do'], 'r-', label='Test (Dropout)', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Overfitting Analysis: Train vs Test Loss', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_statistics(metrics):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nNo Dropout Model:")
    print(f"  Final Test Accuracy: {metrics['acc_no'][-1]:.4f} ({metrics['acc_no'][-1]*100:.2f}%)")
    print(f"  Best Test Accuracy: {max(metrics['acc_no']):.4f} ({max(metrics['acc_no'])*100:.2f}%)")
    print(f"  Final Train Loss: {metrics['train_loss_no'][-1]:.4f}")
    print(f"  Final Test Loss: {metrics['test_loss_no'][-1]:.4f}")
    print(f"  Overfitting Gap: {metrics['test_loss_no'][-1] - metrics['train_loss_no'][-1]:.4f}")
    
    print(f"\nDropout Model (Deterministic):")
    print(f"  Final Test Accuracy: {metrics['acc_do_det'][-1]:.4f} ({metrics['acc_do_det'][-1]*100:.2f}%)")
    print(f"  Best Test Accuracy: {max(metrics['acc_do_det']):.4f} ({max(metrics['acc_do_det'])*100:.2f}%)")
    print(f"  Final Train Loss: {metrics['train_loss_do'][-1]:.4f}")
    print(f"  Final Test Loss: {metrics['test_loss_do'][-1]:.4f}")
    print(f"  Overfitting Gap: {metrics['test_loss_do'][-1] - metrics['train_loss_do'][-1]:.4f}")
    
    print(f"\nDropout Model (MC):")
    print(f"  Final Test Accuracy: {metrics['acc_do_mc'][-1]:.4f} ({metrics['acc_do_mc'][-1]*100:.2f}%)")
    print(f"  Best Test Accuracy: {max(metrics['acc_do_mc']):.4f} ({max(metrics['acc_do_mc'])*100:.2f}%)")
    
    print(f"\nComparisons:")
    print(f"  MC vs Deterministic (final): {(metrics['acc_do_mc'][-1] - metrics['acc_do_det'][-1])*100:+.2f}%")
    print(f"  Dropout vs No Dropout (final): {(metrics['acc_do_det'][-1] - metrics['acc_no'][-1])*100:+.2f}%")
    print(f"  MC vs No Dropout (final): {(metrics['acc_do_mc'][-1] - metrics['acc_no'][-1])*100:+.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Load metrics
    metrics_path = "pickled/mc_dropout/training_metrics.pkl"
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Please run the training script first.")
    else:
        metrics = load_metrics(metrics_path)
        print(f"Loaded metrics from {metrics_path}")
        
        # Print statistics
        print_statistics(metrics)
        
        # Plot results
        plot_all_metrics(metrics, save_path="pickled/mc_dropout/replotted_results.png")

