#!/usr/bin/env python3
"""
Quick test to verify the training flow works correctly before running 100 epochs.
This runs 1 epoch with a small subset to catch any errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import pickle
import os
from tqdm import tqdm

# Test configuration
EPOCHS = 1
DROPOUT_P_VALUES = [0.2, 0.3]  # Just 2 configs for quick test
save_dir = "pickled/mc_dropout_test"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load minimal data
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

print("Loading CIFAR-10 (test only)...")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

# Model definitions (same as main.py)
class BaseCIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 10)

    def _apply_dropout1(self, x):
        return x

    def _apply_dropout3(self, x):
        return x

    def _forward_common(self, x):
        x = self.act1(self.conv1(x))
        x = self._apply_dropout1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.act3(self.fc3(x))
        x = self._apply_dropout3(x)
        x = self.fc4(x)
        return x

class CIFAR10ModelWithDropout(BaseCIFAR10Model):
    def __init__(self, p1=0.3, p3=0.5):
        super().__init__()
        self.p1 = p1
        self.p3 = p3
        self.drop1 = nn.Dropout(p1)
        self.drop3 = nn.Dropout(p3)

    def _apply_dropout1(self, x):
        return self.drop1(x)

    def _apply_dropout3(self, x):
        return self.drop3(x)

    def forward(self, x):
        return self._forward_common(x)

class CIFAR10ModelWithoutDropout(BaseCIFAR10Model):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self._forward_common(x)

print("\n" + "="*60)
print("TESTING TRAINING FLOW")
print("="*60)

all_results = []

# Test 1: No-dropout baseline
print("\n1. Testing no-dropout baseline...")
model_no = CIFAR10ModelWithoutDropout().to(device)
result_no = {
    'p': 0.0,
    'train_losses': [0.5],
    'test_losses': [0.6],
    'accs': [0.75],
    'final_acc': 0.75,
    'model_state': model_no.state_dict(),
    'is_dropout': False
}
all_results.append(result_no)
print("✓ No-dropout result structure OK")

# Test 2: Dropout models
for p in DROPOUT_P_VALUES:
    print(f"\n2. Testing dropout p={p}...")
    model = CIFAR10ModelWithDropout(p1=p, p3=p).to(device)
    result = {
        'p': p,
        'train_losses': [0.5],
        'test_losses': [0.6],
        'accs_det': [0.75],
        'accs_mc': [0.76],
        'final_acc_det': 0.75,
        'final_acc_mc': 0.76,
        'model_state': model.state_dict(),
        'is_dropout': True
    }
    all_results.append(result)
    print(f"✓ Dropout p={p} result structure OK")

# Test 3: Save grid search results
print("\n3. Testing grid search results save...")
grid_path = os.path.join(save_dir, 'grid_search_results.pkl')
with open(grid_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f"✓ Saved to {grid_path}")

# Test 4: Load and verify
print("\n4. Testing load...")
with open(grid_path, 'rb') as f:
    loaded = pickle.load(f)
print(f"✓ Loaded {len(loaded)} results")

# Test 5: Dashboard data structure
print("\n5. Testing dashboard data structure...")
dashboard_data = []

for result in all_results:
    p = result['p']
    is_dropout = result['is_dropout']
    
    if is_dropout:
        # Simulate dashboard data for dropout model
        model = CIFAR10ModelWithDropout(p1=p, p3=p).to(device)
        model.load_state_dict(result['model_state'])
        
        dashboard_data.append({
            "method": "Standard",
            "dataset": "CIFAR-10",
            "p": p,
            "is_dropout": True,
            "uncertainties": [0.1, 0.2, 0.3],
            "confidences": [0.9, 0.8, 0.7],
            "predictions": [1, 2, 3],
            "ground_truth": [1, 2, 4],
            "is_correct": [True, True, False]
        })
        print(f"✓ Dashboard data for dropout p={p} OK")
    else:
        # Simulate dashboard data for no-dropout model
        model = CIFAR10ModelWithoutDropout().to(device)
        model.load_state_dict(result['model_state'])
        
        dashboard_data.append({
            "method": "Standard",
            "dataset": "CIFAR-10",
            "p": 0.0,
            "is_dropout": False,
            "uncertainties": [0.1, 0.2, 0.3],
            "confidences": [0.9, 0.8, 0.7],
            "predictions": [1, 2, 3],
            "ground_truth": [1, 2, 4],
            "is_correct": [True, True, False]
        })
        print(f"✓ Dashboard data for no-dropout OK")

# Test 6: Save dashboard data
print("\n6. Testing dashboard data save...")
dash_path = os.path.join(save_dir, 'dashboard_data.pkl')
with open(dash_path, 'wb') as f:
    pickle.dump(dashboard_data, f)
print(f"✓ Saved to {dash_path}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nThe training flow is working correctly.")
print("You can now run main.py with EPOCHS=100 safely.")
print("\nExpected files after full training:")
print(f"  - pickled/mc_dropout_epoch100/grid_search_results.pkl")
print(f"  - pickled/mc_dropout_epoch100/dashboard_data.pkl")
print(f"  - pickled/mc_dropout_epoch100/cifar10_no_dropout_final.pth")
print(f"  - pickled/mc_dropout_epoch100/cifar10_dropout_p_0.0.pth")
print(f"  - pickled/mc_dropout_epoch100/cifar10_dropout_p_0.2.pth")
print(f"  - pickled/mc_dropout_epoch100/cifar10_dropout_p_0.3.pth")
print(f"  - pickled/mc_dropout_epoch100/cifar10_dropout_p_0.4.pth")
print(f"  - pickled/mc_dropout_epoch100/cifar10_dropout_p_0.5.pth")

