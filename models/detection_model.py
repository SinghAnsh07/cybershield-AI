import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets, weights=None):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if weights is not None:
            focal_loss = focal_loss * weights
        return focal_loss.mean()

class LSTMDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.NUM_FEATURES,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, weights):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.weights = torch.FloatTensor(weights).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.weights[idx]

class DetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMDetector().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LSTM_LEARNING_RATE)
        self.criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
        
    def _create_sequences(self, features, labels, entity_ids):
        sequences = []
        seq_labels = []
        weights = []
        
        feat_vals = features.values if hasattr(features, 'values') else features
        lab_vals = labels.values if hasattr(labels, 'values') else labels
        ent_vals = entity_ids.values if hasattr(entity_ids, 'values') else entity_ids
        
        # Convert string labels to binary
        if lab_vals.dtype.kind in ('U', 'S', 'O'):  # string types
            lab_vals = (lab_vals != config.LABEL_NORMAL).astype(int)
        
        for eid in np.unique(ent_vals):
            mask = ent_vals == eid
            ent_feats = feat_vals[mask]
            ent_labs = lab_vals[mask]
            
            n_events = len(ent_feats)
            if n_events < config.SEQUENCE_LENGTH:
                continue
                
            for i in range(n_events - config.SEQUENCE_LENGTH + 1):
                seq = ent_feats[i:i+config.SEQUENCE_LENGTH]
                label = ent_labs[i+config.SEQUENCE_LENGTH-1]
                age = (n_events - (i+config.SEQUENCE_LENGTH))
                weight = config.DRIFT_DECAY_FACTOR ** age
                
                sequences.append(seq)
                seq_labels.append(label)
                weights.append(weight)
                
        if not sequences:
            return np.array([]), np.array([]), np.array([])
        return np.array(sequences), np.array(seq_labels), np.array(weights)
        
    def fit(self, features_df, labels, entity_ids):
        X_seq, y_seq, w_seq = self._create_sequences(features_df, labels, entity_ids)
        if len(X_seq) == 0:
            return
            
        n_samples = len(X_seq)
        val_size = int(n_samples * config.VALIDATION_SPLIT_RATIO)
        indices = np.random.permutation(n_samples)
        
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        train_dataset = SequenceDataset(X_seq[train_idx], y_seq[train_idx], w_seq[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=True)
        
        self.model.train()
        for epoch in range(config.LSTM_EPOCHS):
            for seq, label, w in train_loader:
                seq, label, w = seq.to(self.device), label.to(self.device), w.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(seq)
                loss = self.criterion(out, label, w)
                loss.backward()
                self.optimizer.step()
                
    def predict(self, features, entity_ids):
        feat_vals = features.values if hasattr(features, 'values') else features
        ent_vals = entity_ids.values if hasattr(entity_ids, 'values') else entity_ids
        
        n_samples = len(feat_vals)
        risk_scores = np.zeros(n_samples)
        anomaly_flags = np.zeros(n_samples, dtype=int)
        
        self.model.eval()
        with torch.no_grad():
            for eid in np.unique(ent_vals):
                idx = np.where(ent_vals == eid)[0]
                ent_feats = feat_vals[idx]
                n_events = len(ent_feats)
                
                if n_events == 0:
                    continue
                    
                seqs = []
                for i in range(n_events):
                    if i < config.SEQUENCE_LENGTH:
                        pad_len = config.SEQUENCE_LENGTH - i - 1
                        pad = np.repeat(ent_feats[0:1], pad_len, axis=0) if pad_len > 0 else np.empty((0, ent_feats.shape[1]))
                        seq = np.vstack((pad, ent_feats[0:i+1]))
                    else:
                        seq = ent_feats[i-config.SEQUENCE_LENGTH+1:i+1]
                    seqs.append(seq)
                    
                seqs = torch.FloatTensor(np.array(seqs)).to(self.device)
                batch_size = 512
                scores = []
                for b in range(0, len(seqs), batch_size):
                    batch_seqs = seqs[b:b+batch_size]
                    out = self.model(batch_seqs).cpu().numpy().flatten()
                    scores.extend(out)
                    
                risk_scores[idx] = scores
                anomaly_flags[idx] = (np.array(scores) > 0.5).astype(int)
                
        return anomaly_flags, risk_scores

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path.with_suffix('.pt'))
        
    def load(self, path):
        path = Path(path)
        self.model.load_state_dict(torch.load(path.with_suffix('.pt'), map_location=self.device))
