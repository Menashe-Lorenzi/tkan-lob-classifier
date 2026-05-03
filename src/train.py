import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy

class LOBSequenceDataset(Dataset):
    """Creates sliding windows of sequence_length, strictly within the same trading day."""
    def __init__(self, X, y, dates, sequence_length):
        self.X = X
        self.y = y
        self.dates = dates
        self.sequence_length = sequence_length
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        valid = []
        # Iterate through the data, stopping before the end
        for i in range(len(self.X) - self.sequence_length):
            # Check if the start of the sequence and the end of the sequence share the same date
            if self.dates[i] == self.dates[i + self.sequence_length - 1]:
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map the continuous index to our filtered, valid indices
        real_idx = self.valid_indices[idx]
        
        x_seq = self.X[real_idx : real_idx + self.sequence_length]
        y_label = self.y[real_idx + self.sequence_length - 1]
        
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

def assign_classes(df, q_lower, q_upper):
    """Assigns target classes based on calculated quantiles."""
    conditions = [
        (df['next_ofi'] < q_lower),
        (df['next_ofi'] >= q_lower) & (df['next_ofi'] <= q_upper),
        (df['next_ofi'] > q_upper)
    ]
    df['target'] = np.select(conditions, [0, 1, 2], default=1)
    return df

def train_single_fold(model, train_loader, test_loader, device, num_epochs=200, patience=2):
    """Handles the PyTorch training loop with early stopping."""
    manual_weights = torch.tensor([3.0, 1.0, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    
    best_val_loss = float('inf')
    best_val_f1, best_val_acc = 0.0, 0.0
    epochs_no_improve = 0
    history = {'train': [], 'val': []}
    
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(test_loader.dataset)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        val_acc = accuracy_score(all_targets, all_preds)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict()) # Save weights when improving
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # LOAD THE BEST WEIGHTS BEFORE RETURNING!
    model.load_state_dict(best_model_weights)

    return best_val_f1, best_val_acc, model, history

def run_day_forward_validation(df, feature_cols, model_class, model_kwargs, device, train_window_days=5, seq_length=5, num_epochs=200):
    """Executes Day-Forward Validation, ensuring train/test splits occur cleanly at midnight."""
    
    # Create a clean date column for logic
    df['date_only'] = df.index.date
    unique_dates = df['date_only'].unique()
    
    fold_f1_scores, fold_acc_scores = [], []
    
    print("=" * 60)
    print(f"Starting Day-Forward Validation ({model_class.__name__})")
    
    final_history_dict = None
    final_trained_model = None

    # Start iterating from the day after our initial training window
    fold = 1
    for i in range(train_window_days, len(unique_dates)):
        train_dates = unique_dates[:i] # Train on all days up to day i
        test_date = unique_dates[i]    # Test strictly on day i
        
        print(f"\n--- FOLD {fold} | Train: {len(train_dates)} days | Test: {test_date} ---")
        
        train_df = df[df['date_only'].isin(train_dates)].copy()
        test_df = df[df['date_only'] == test_date].copy()
        
        # Calculate thresholds strictly on training data
        q_lower = train_df['next_ofi'].quantile(0.2)
        q_upper = train_df['next_ofi'].quantile(0.8)
        
        train_df = assign_classes(train_df, q_lower, q_upper)
        test_df = assign_classes(test_df, q_lower, q_upper)
        
        X_train, y_train = train_df[feature_cols].values, train_df['target'].values
        dates_train = train_df['date_only'].values
        
        X_test, y_test = test_df[feature_cols].values, test_df['target'].values
        dates_test = test_df['date_only'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Pass the dates array into our new dataset to prevent overnight bleed
        train_loader = DataLoader(LOBSequenceDataset(X_train_scaled, y_train, dates_train, seq_length), batch_size=2048, shuffle=False)
        test_loader = DataLoader(LOBSequenceDataset(X_test_scaled, y_test, dates_test, seq_length), batch_size=2048, shuffle=False)
        
        # Initialize the model just once here
        model = model_class(**model_kwargs).to(device)
        
        val_f1, val_acc, trained_model, history_dict = train_single_fold(model, train_loader, test_loader, device, num_epochs)
        
        final_trained_model = trained_model
        final_history_dict = history_dict
        
        print(f"Fold {fold} | Best F1: {val_f1:.4f} | Best Acc: {val_acc:.4f}")
        fold_f1_scores.append(val_f1)
        fold_acc_scores.append(val_acc)
        fold += 1

    print("\n" + "=" * 60)
    print(f"🏆 {model_class.__name__} RESULTS 🏆")
    print(f"Macro-F1: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
    print(f"Accuracy: {np.mean(fold_acc_scores):.4f} (+/- {np.std(fold_acc_scores):.4f})")
    print("=" * 60)

    # Make sure this return is perfectly aligned with the prints!
    return final_trained_model, fold_f1_scores, fold_acc_scores, final_history_dict


def get_model_predictions(model, data_loader, device):
    """Runs inference on a DataLoader and returns (targets, predictions) arrays."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    return np.array(all_targets), np.array(all_preds)