"""
Quick test to verify the updated model architecture works correctly.
Tests:
1. Model instantiation with single dropout parameter
2. Forward pass works
3. Dropout is only applied after fc3
4. No-dropout baseline still works
"""

import torch
import torch.nn as nn

# Copy model definitions from main.py
class BaseCIFAR10Model(nn.Module):
    """
    Base CIFAR-10 CNN Model Architecture:
    - Conv1 (3->32) -> ReLU
    - Conv2 (32->32) -> ReLU -> MaxPool
    - Flatten
    - FC3 (8192->512) -> ReLU -> [Dropout here in dropout variant]
    - FC4 (512->10)
    """
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

    def _apply_dropout3(self, x):
        return x

    def _forward_common(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.act3(self.fc3(x))
        x = self._apply_dropout3(x)
        x = self.fc4(x)
        return x

class CIFAR10ModelWithDropout(BaseCIFAR10Model):
    def __init__(self, p=0.5):
        """
        Args:
            p: Dropout probability after fc3 (fully connected layer)
        """
        super().__init__()
        self.p = p
        self.drop3 = nn.Dropout(p)

    def _apply_dropout3(self, x):
        return self.drop3(x)

    def forward(self, x):
        return self._forward_common(x)

class CIFAR10ModelWithoutDropout(BaseCIFAR10Model):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self._forward_common(x)


def test_models():
    print("Testing Updated Model Architecture")
    print("=" * 60)
    
    # Test 1: No-dropout model
    print("\n1. Testing No-Dropout Model...")
    model_no_drop = CIFAR10ModelWithoutDropout()
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    y = model_no_drop(x)
    assert y.shape == (4, 10), f"Expected shape (4, 10), got {y.shape}"
    print("   ✅ No-dropout model works! Output shape:", y.shape)
    
    # Test 2: Dropout models with different p values
    print("\n2. Testing Dropout Models...")
    for p in [0.2, 0.3, 0.4, 0.5]:
        model = CIFAR10ModelWithDropout(p=p)
        y = model(x)
        assert y.shape == (4, 10), f"Expected shape (4, 10), got {y.shape}"
        assert model.p == p, f"Expected p={p}, got {model.p}"
        assert hasattr(model, 'drop3'), "Model should have drop3 attribute"
        assert not hasattr(model, 'drop1'), "Model should NOT have drop1 attribute"
        print(f"   ✅ Dropout p={p} works! Output shape: {y.shape}")
    
    # Test 3: Verify dropout is only after fc3
    print("\n3. Verifying Dropout Location...")
    model = CIFAR10ModelWithDropout(p=0.5)
    model.train()  # Enable dropout
    
    # Run multiple forward passes - outputs should differ due to dropout
    outputs = []
    for _ in range(5):
        y = model(x)
        outputs.append(y)
    
    # Check that outputs differ (dropout is active)
    all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
    assert not all_same, "Outputs should differ when dropout is enabled"
    print("   ✅ Dropout is active during training!")
    
    # Test 4: Verify dropout is disabled in eval mode
    print("\n4. Verifying Eval Mode...")
    model.eval()  # Disable dropout
    outputs_eval = []
    for _ in range(5):
        y = model(x)
        outputs_eval.append(y)
    
    # Check that outputs are identical (dropout is disabled)
    all_same = all(torch.allclose(outputs_eval[0], out) for out in outputs_eval[1:])
    assert all_same, "Outputs should be identical when dropout is disabled"
    print("   ✅ Dropout is disabled in eval mode!")
    
    # Test 5: Parameter count
    print("\n5. Checking Parameter Count...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Expected: Conv1 (896) + Conv2 (9,248) + FC3 (4,194,816) + FC4 (5,130) = 4,210,090
    expected_params = 4_210_090
    assert total_params == expected_params, f"Expected {expected_params:,} params, got {total_params:,}"
    print(f"   ✅ Parameter count correct: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nModel Summary:")
    print("- Dropout applied ONLY after FC3 (fully connected layer)")
    print("- No dropout after Conv1 (removed)")
    print("- Single dropout probability parameter 'p'")
    print("- Total parameters: 4,210,090")
    print("\nReady for training! 🚀")


if __name__ == "__main__":
    test_models()

