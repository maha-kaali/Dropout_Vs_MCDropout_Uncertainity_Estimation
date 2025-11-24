# CIFAR10 CNN Model with Enhanced Training & Dashboard Data Collection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import time
import pickle
import os
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# DIRECTORY PATH => must fill new to avoid overwriting #[can_change]
save_dir = "pickled/mc_dropout_epoch100"
os.makedirs(save_dir, exist_ok=True)

train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

print("Loading CIFAR-10...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

print("Loading SVHN (Out-of-Distribution)...")
try:
    svhn_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_tf)
    svhn_loader = torch.utils.data.DataLoader(svhn_set, batch_size=256, shuffle=False, num_workers=0)
    has_svhn = True
except Exception as e:
    print(f"Could not load SVHN: {e}. OOD steps will be skipped.")
    has_svhn = False

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

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

loss_fn = nn.CrossEntropyLoss()



def train_epoch(model, opt):
    model.train()
    loss_sum = 0
    n = 0
    pbar = tqdm(train_loader, desc="Training", leave=True)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * xb.size(0)
        n += xb.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return loss_sum / n



@torch.no_grad()
def test_acc_and_loss(model, loader=test_loader):
    model.eval()
    correct = 0
    n = 0
    loss_sum = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        loss_sum += loss.item() * xb.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += xb.size(0)
    return correct / n, loss_sum / n



@torch.no_grad()
def test_acc_mc(model, T=20):
    model.train()  # Keep dropout enabled
    preds = []
    for t in range(T):
        all_preds = []
        for xb, _ in test_loader:
            xb = xb.to(device)
            out = model(xb)
            out = F.softmax(out, dim=1)
            all_preds.append(out.cpu().numpy())
        preds.append(np.concatenate(all_preds))
    preds = np.stack(preds, axis=0)
    mean_probs = preds.mean(axis=0)
    pred = mean_probs.argmax(axis=1)
    ys = np.concatenate([y.numpy() for _, y in test_loader])
    return (pred == ys).mean()


# Grid Search
ENABLE_GRID_SEARCH = True

DROPOUT_P_VALUES = [0.2, 0.3, 0.4, 0.5]  # Single dropout probability for both layers #[can_change]


if __name__ == '__main__':
    EPOCHS = 100 # [can_change]

    # Storage for all experiments
    all_results = []

    if ENABLE_GRID_SEARCH:
        print(f"\n{'='*60}")
        print(f"GRID SEARCH MODE: Testing {len(DROPOUT_P_VALUES)} dropout configurations + 1 no-dropout baseline")
        print(f"Total models to train: {len(DROPOUT_P_VALUES) + 1}")
        print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print(f"Training NO DROPOUT baseline")
        print(f"{'='*60}\n")

        model_no = CIFAR10ModelWithoutDropout().to(device)
        optimizer_no = torch.optim.Adam(model_no.parameters(), lr=1e-3)

        train_losses_no, test_losses_no, accs_no = [], [], []

        for ep in range(1, EPOCHS+1):
            t0 = time.time()

            l_train = train_epoch(model_no, optimizer_no)
            a_test, l_test = test_acc_and_loss(model_no)

            train_losses_no.append(l_train)
            test_losses_no.append(l_test)
            accs_no.append(a_test)

            elapsed = time.time() - t0
            print(f"Epoch {ep:2d}/{EPOCHS} | Acc: {a_test:.3f} | Train Loss: {l_train:.4f} | Test Loss: {l_test:.4f} | Time: {elapsed:.1f}s")

        # Store no-dropout results
        no_dropout_result = {
            'p': 0.0,
            'train_losses': train_losses_no,
            'test_losses': test_losses_no,
            'accs': accs_no,
            'final_acc': accs_no[-1],
            'model_state': model_no.state_dict(),
            'is_dropout': False
        }
        all_results.append(no_dropout_result)

        torch.save(model_no.state_dict(), os.path.join(save_dir, 'cifar10_no_dropout_final.pth'))
        print(f"No-dropout model saved")

        # start grid search

        for p in DROPOUT_P_VALUES:
            print(f"\n{'='*60}")
            print(f"Training with dropout p={p} (after fc3 only)")
            print(f"{'='*60}\n")

            model = CIFAR10ModelWithDropout(p=p).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses, test_losses, accs_det, accs_mc = [], [], [], []

            # train for both mc and non mc
            for ep in range(1, EPOCHS+1):
                t0 = time.time()

                # Train
                l_train = train_epoch(model, optimizer)
                a_det, l_test = test_acc_and_loss(model)
                a_mc = test_acc_mc(model, T=5)

                train_losses.append(l_train)
                test_losses.append(l_test)
                accs_det.append(a_det)
                accs_mc.append(a_mc)

                elapsed = time.time() - t0
                print(f"Epoch {ep:2d}/{EPOCHS} | Det Acc: {a_det:.3f} | MC Acc: {a_mc:.3f} | Train Loss: {l_train:.4f} | Test Loss: {l_test:.4f} | Time: {elapsed:.1f}s")

            result = {
                'p': p,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accs_det': accs_det,
                'accs_mc': accs_mc,
                'final_acc_det': accs_det[-1],
                'final_acc_mc': accs_mc[-1],
                'model_state': model.state_dict(),
                'is_dropout': True
            }
            all_results.append(result)

            model_filename = f'cifar10_dropout_p_{p}.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
            print(f"Model saved: {model_filename}")

        grid_results_path = os.path.join(save_dir, 'grid_search_results.pkl')
        with open(grid_results_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n{'='*60}")
        print(f"Grid search complete! Results saved to: {grid_results_path}")
        print(f"{'='*60}\n")

     
        print("\nGRID SEARCH SUMMARY:")
        print(f"{'Model':<20} {'p':<6} {'Acc':<10} {'MC Acc':<10} {'Improvement':<12}")
        print("-" * 60)

        no_drop = [r for r in all_results if not r['is_dropout']][0]
        print(f"{'No Dropout':<20} {no_drop['p']:<6.1f} {no_drop['final_acc']:<10.4f} {'N/A':<10} {'N/A':<12}")

        for r in [r for r in all_results if r['is_dropout']]:
            improvement = (r['final_acc_mc'] - r['final_acc_det']) * 100
            print(f"{'Dropout':<20} {r['p']:<6.1f} {r['final_acc_det']:<10.4f} {r['final_acc_mc']:<10.4f} {improvement:<12.2f}%")

        # Find best configuration
        dropout_results = [r for r in all_results if r['is_dropout']]
        if dropout_results:
            best_mc = max(dropout_results, key=lambda x: x['final_acc_mc'])
            print(f"\nBest MC Accuracy: p={best_mc['p']} -> {best_mc['final_acc_mc']:.4f}")

    else:
        print(f"\n{'='*60}")
        print(f"SINGLE MODEL MODE: Training with default dropout (p=0.3)")
        print(f"{'='*60}\n")

        print("Training NO DROPOUT baseline...")
        model_no = CIFAR10ModelWithoutDropout().to(device)
        opt_no = torch.optim.Adam(model_no.parameters(), lr=1e-3)

        train_loss_no, test_loss_no, acc_no = [], [], []

        for ep in range(1, EPOCHS+1):
            t0 = time.time()
            l_train = train_epoch(model_no, opt_no)
            a_test, l_test = test_acc_and_loss(model_no)
            train_loss_no.append(l_train)
            test_loss_no.append(l_test)
            acc_no.append(a_test)
            elapsed = time.time() - t0
            print(f"Epoch {ep:2d}/{EPOCHS} | Acc: {a_test:.3f} | Train Loss: {l_train:.4f} | Test Loss: {l_test:.4f} | Time: {elapsed:.1f}s")

        no_dropout_result = {
            'p': 0.0,
            'train_losses': train_loss_no,
            'test_losses': test_loss_no,
            'accs': acc_no,
            'final_acc': acc_no[-1],
            'model_state': model_no.state_dict(),
            'is_dropout': False
        }
        all_results.append(no_dropout_result)

        print("\nTraining WITH DROPOUT (p=0.3)...")
        model_do = CIFAR10ModelWithDropout(p1=0.3, p3=0.3).to(device)
        opt_do = torch.optim.Adam(model_do.parameters(), lr=1e-3)

        train_loss_do, test_loss_do, acc_do_det, acc_do_mc = [], [], [], []

        for ep in range(1, EPOCHS+1):
            t0 = time.time()
            l2_train = train_epoch(model_do, opt_do)
            a2, l2_test = test_acc_and_loss(model_do)
            a3 = test_acc_mc(model_do, T=5)

            train_loss_do.append(l2_train)
            test_loss_do.append(l2_test)
            acc_do_det.append(a2)
            acc_do_mc.append(a3)

            elapsed = time.time() - t0
            print(f"Epoch {ep:2d}/{EPOCHS} | Det Acc: {a2:.3f} | MC Acc: {a3:.3f} | Train Loss: {l2_train:.4f} | Test Loss: {l2_test:.4f} | Time: {elapsed:.1f}s")

        dropout_result = {
            'p': 0.3,
            'train_losses': train_loss_do,
            'test_losses': test_loss_do,
            'accs_det': acc_do_det,
            'accs_mc': acc_do_mc,
            'final_acc_det': acc_do_det[-1],
            'final_acc_mc': acc_do_mc[-1],
            'model_state': model_do.state_dict(),
            'is_dropout': True
        }
        all_results.append(dropout_result)

        torch.save(model_no.state_dict(), os.path.join(save_dir, 'cifar10_no_dropout_final.pth'))
        torch.save(model_do.state_dict(), os.path.join(save_dir, 'cifar10_dropout_p_0.3.pth'))
        print("\nTraining complete. Models saved.")

    
    # Streamlit dashboard data collection 


    print(f"\n{'='*60}")
    print("STARTING DATA COLLECTION FOR DASHBOARD")
    print(f"{'='*60}")

    dashboard_data = []

    def collect_inference_data(model, loader, dataset_name, method, p=None, is_dropout=True, T=1):
        """
        Runs inference and collects:
        - Probabilities (Confidences)
        - Entropy (Uncertainty)
        - Correctness
        """
        config_str = f"p={p}" if p is not None else "default"
        print(f"Collecting: {dataset_name} | {method} | {config_str} | T={T} ...")

        if method == "Standard":
            model.eval() # Dropout OFF
        else:
            model.train() # Dropout ON (MC Mode)

        all_probs_per_pass = []
        all_targets = []

        # 1. Run Forward Passes
        for t in range(T):
            pass_probs = []
            pass_targets = []
            with torch.no_grad():
                for xb, yb in tqdm(loader, leave=False, desc=f"Pass {t+1}/{T}"):
                    xb = xb.to(device)
                    out = model(xb)
                    probs = F.softmax(out, dim=1) # [batch, 10]
                    pass_probs.append(probs.cpu().numpy())
                    pass_targets.append(yb.numpy())

            all_probs_per_pass.append(np.concatenate(pass_probs))
            if t == 0: all_targets = np.concatenate(pass_targets)

        
        stacked_probs = np.stack(all_probs_per_pass)

        # Average over T passes (for Standard, T=1, so mean is just the value)
        mean_probs = np.mean(stacked_probs, axis=0)
     
        predictions = np.argmax(mean_probs, axis=1)

        # A. CONFIDENCE: Max probability for the predicted class
        confidences = np.max(mean_probs, axis=1)

        # B. UNCERTAINTY (Entropy): -sum(p * log(p))
        # Add epsilon to avoid log(0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        # C. CORRECTNESS
        is_correct = (predictions == all_targets)

        return {
            "method": method,
            "dataset": dataset_name,
            "p": p if p is not None else 0.0,
            "is_dropout": is_dropout,
            "uncertainties": entropy.tolist(),     
            "confidences": confidences.tolist(),
            "predictions": predictions.tolist(),
            "ground_truth": all_targets.tolist(),
            "is_correct": is_correct.tolist()
        }


    for result in all_results:
        p = result['p']
        is_dropout = result['is_dropout']

        if is_dropout:
            print(f"\nCollecting dashboard data for dropout p={p} (after fc3 only)...")

            model = CIFAR10ModelWithDropout(p=p).to(device)
            model.load_state_dict(result['model_state'])

            # 1. Standard on CIFAR-10
            data_std = collect_inference_data(model, test_loader, "CIFAR-10", "Standard", p=p, is_dropout=True, T=1)
            dashboard_data.append(data_std)

            # 2. MC Dropout on CIFAR-10
            data_mc = collect_inference_data(model, test_loader, "CIFAR-10", "MC Dropout", p=p, is_dropout=True, T=30)
            dashboard_data.append(data_mc)

            if has_svhn:
                # 3. Standard on SVHN (OOD)
                data_ood_std = collect_inference_data(model, svhn_loader, "SVHN", "Standard", p=p, is_dropout=True, T=1)
                dashboard_data.append(data_ood_std)

                # 4. MC Dropout on SVHN (OOD)
                data_ood_mc = collect_inference_data(model, svhn_loader, "SVHN", "MC Dropout", p=p, is_dropout=True, T=30)
                dashboard_data.append(data_ood_mc)
        else:
            print(f"\nCollecting dashboard data for no-dropout baseline...")
            model = CIFAR10ModelWithoutDropout().to(device)
            model.load_state_dict(result['model_state'])

            # Only standard inference for no-dropout (no MC)
            # 1. Standard on CIFAR-10
            data_std = collect_inference_data(model, test_loader, "CIFAR-10", "Standard", p=0.0, is_dropout=False, T=1)
            dashboard_data.append(data_std)

            if has_svhn:
                # 2. Standard on SVHN (OOD)
                data_ood_std = collect_inference_data(model, svhn_loader, "SVHN", "Standard", p=0.0, is_dropout=False, T=1)
                dashboard_data.append(data_ood_std)


    data_path = os.path.join(save_dir, 'dashboard_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(dashboard_data, f)

    print(f"\nSUCCESS! Dashboard data saved to: {data_path}")
    print(f"Contains {len(dashboard_data)} experiment records.")