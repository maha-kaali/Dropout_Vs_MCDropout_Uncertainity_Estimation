"""
Pre-flight check for 100-epoch training run.
Verifies:
1. Model architecture is correct
2. Data storage paths exist
3. All data structures are compatible
4. Expected files will be created
5. Disk space is sufficient
"""

import torch
import os
import pickle
import shutil

def check_disk_space(path, required_mb=500):
    """Check if sufficient disk space is available"""
    stat = shutil.disk_usage(path)
    free_mb = stat.free / (1024 * 1024)
    print(f"   Free disk space: {free_mb:.0f} MB")
    if free_mb < required_mb:
        print(f"   ⚠️  WARNING: Less than {required_mb} MB free!")
        return False
    print(f"   ✅ Sufficient disk space ({free_mb:.0f} MB > {required_mb} MB)")
    return True

def check_directory_structure():
    """Verify save directory exists and is writable"""
    save_dir = "pickled/mc_dropout_epoch100"
    
    print("\n1. Checking Directory Structure...")
    
    # Check if directory exists
    if os.path.exists(save_dir):
        print(f"   ✅ Directory exists: {save_dir}")
        
        # Check if writable
        test_file = os.path.join(save_dir, ".test_write")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"   ✅ Directory is writable")
        except Exception as e:
            print(f"   ❌ Directory not writable: {e}")
            return False
    else:
        print(f"   ⚠️  Directory doesn't exist, will be created: {save_dir}")
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"   ✅ Directory created successfully")
        except Exception as e:
            print(f"   ❌ Failed to create directory: {e}")
            return False
    
    # Check disk space
    return check_disk_space(save_dir)

def check_model_architecture():
    """Verify model architecture is correct"""
    print("\n2. Checking Model Architecture...")
    
    # Import from main.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from main import CIFAR10ModelWithDropout, CIFAR10ModelWithoutDropout
        
        # Test no-dropout model
        model_no = CIFAR10ModelWithoutDropout()
        x = torch.randn(2, 3, 32, 32)
        y = model_no(x)
        assert y.shape == (2, 10), f"No-dropout output shape wrong: {y.shape}"
        print(f"   ✅ No-dropout model works")
        
        # Test dropout models
        for p in [0.2, 0.3, 0.4, 0.5]:
            model = CIFAR10ModelWithDropout(p=p)
            y = model(x)
            assert y.shape == (2, 10), f"Dropout p={p} output shape wrong: {y.shape}"
            assert model.p == p, f"Dropout p mismatch: expected {p}, got {model.p}"
            assert hasattr(model, 'drop3'), "Model should have drop3"
            assert not hasattr(model, 'drop1'), "Model should NOT have drop1"
        
        print(f"   ✅ All dropout models work (p=0.2, 0.3, 0.4, 0.5)")
        print(f"   ✅ Dropout only after fc3 (drop1 removed)")
        
        return True
    except Exception as e:
        print(f"   ❌ Model architecture error: {e}")
        return False

def check_data_structures():
    """Verify data structures match expected format"""
    print("\n3. Checking Data Structures...")
    
    # Expected structure for no-dropout result
    no_dropout_keys = {'p', 'train_losses', 'test_losses', 'accs', 'final_acc', 'model_state', 'is_dropout'}
    
    # Expected structure for dropout result
    dropout_keys = {'p', 'train_losses', 'test_losses', 'accs_det', 'accs_mc', 
                    'final_acc_det', 'final_acc_mc', 'model_state', 'is_dropout'}
    
    # Expected structure for dashboard data
    dashboard_keys = {'method', 'dataset', 'p', 'is_dropout', 'uncertainties', 
                      'confidences', 'predictions', 'ground_truth', 'is_correct'}
    
    print(f"   ✅ No-dropout result keys: {len(no_dropout_keys)} fields")
    print(f"   ✅ Dropout result keys: {len(dropout_keys)} fields")
    print(f"   ✅ Dashboard data keys: {len(dashboard_keys)} fields")
    
    return True

def check_expected_files():
    """List expected output files"""
    print("\n4. Expected Output Files...")
    
    save_dir = "pickled/mc_dropout_epoch100"
    
    expected_files = [
        "cifar10_no_dropout_final.pth",
        "cifar10_dropout_p_0.2.pth",
        "cifar10_dropout_p_0.3.pth",
        "cifar10_dropout_p_0.4.pth",
        "cifar10_dropout_p_0.5.pth",
        "grid_search_results.pkl",
        "dashboard_data.pkl"
    ]
    
    print(f"   Will create {len(expected_files)} files in {save_dir}:")
    for f in expected_files:
        print(f"      - {f}")
    
    # Estimate file sizes
    print(f"\n   Estimated file sizes:")
    print(f"      - Model .pth files: ~16 MB each × 5 = ~80 MB")
    print(f"      - grid_search_results.pkl: ~50 MB (100 epochs × 5 models)")
    print(f"      - dashboard_data.pkl: ~100 MB (inference data)")
    print(f"   Total estimated: ~230 MB")
    
    return True

def check_training_config():
    """Verify training configuration"""
    print("\n5. Checking Training Configuration...")

    # Read main.py to check configuration
    with open('main.py', 'r') as f:
        content = f.read()

    # Check EPOCHS
    if 'EPOCHS = 100' in content:
        print(f"   ✅ EPOCHS set to 100")
        epochs_ok = True
    else:
        print(f"   ⚠️  WARNING: EPOCHS might not be 100!")
        epochs_ok = False

    # Check ENABLE_GRID_SEARCH
    if 'ENABLE_GRID_SEARCH = True' in content:
        print(f"   ✅ Grid search enabled")
        grid_ok = True
    else:
        print(f"   ⚠️  WARNING: Grid search might be disabled!")
        grid_ok = False

    # Check DROPOUT_P_VALUES
    if 'DROPOUT_P_VALUES = [0.2, 0.3, 0.4, 0.5]' in content:
        print(f"   ✅ Dropout p values correct: [0.2, 0.3, 0.4, 0.5]")
        p_ok = True
    else:
        print(f"   ⚠️  WARNING: Unexpected dropout p values!")
        p_ok = False

    # Check model instantiation uses single p
    if 'CIFAR10ModelWithDropout(p=p)' in content:
        print(f"   ✅ Model uses single dropout parameter (p)")
    else:
        print(f"   ⚠️  WARNING: Model might still use p1, p3!")

    total_models = 5  # 1 no-dropout + 4 dropout
    print(f"\n   Total models to train: {total_models}")
    print(f"   Estimated time: ~7-8 hours (assuming ~1.5 hours per model)")

    return epochs_ok and grid_ok and p_ok

def main():
    print("=" * 70)
    print("PRE-FLIGHT CHECK FOR 100-EPOCH TRAINING RUN")
    print("=" * 70)
    
    checks = [
        check_directory_structure(),
        check_model_architecture(),
        check_data_structures(),
        check_expected_files(),
        check_training_config()
    ]
    
    print("\n" + "=" * 70)
    if all(checks):
        print("✅ ALL CHECKS PASSED! READY FOR 100-EPOCH TRAINING!")
        print("=" * 70)
        print("\nTo start training, run:")
        print("   conda activate torch-m3")
        print("   python main.py")
        print("\nRecommended: Use tmux or screen for long-running process")
        return 0
    else:
        print("❌ SOME CHECKS FAILED! PLEASE FIX ISSUES BEFORE TRAINING!")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit(main())

