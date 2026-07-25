import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import scipy.spatial.distance as distance

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class StatisticalProfiler:
    def __init__(self):
        self.entity_stats = {}
    
    def fit(self, df, features_df):
        if isinstance(features_df, np.ndarray):
            features_df = pd.DataFrame(features_df, columns=config.FEATURE_NAMES)
        features_df = features_df.copy()
        features_df['entity_id'] = df['entity_id'].values
        
        for eid, group in features_df.groupby('entity_id'):
            group_features = group.drop(columns=['entity_id'])
            mean = group_features.mean().values
            std = group_features.std().values
            cov = group_features.cov().values
            if np.isnan(cov).all():
                cov = np.eye(len(mean))
            else:
                cov = np.nan_to_num(cov)
                # add regularization
                cov += np.eye(len(mean)) * 1e-5
            try:
                inv_cov = np.linalg.inv(cov)
            except:
                inv_cov = np.eye(len(mean))
            
            self.entity_stats[eid] = {
                'mean': mean,
                'std': std,
                'inv_cov': inv_cov
            }
            
    def score(self, entity_id, feature_vector):
        if entity_id not in self.entity_stats:
            return 0.5
        stats = self.entity_stats[entity_id]
        diff = feature_vector - stats['mean']
        mah_dist = distance.mahalanobis(feature_vector, stats['mean'], stats['inv_cov'])
        score = 1.0 - np.exp(-mah_dist)
        return score
        
    def get_entity_stats(self, entity_id):
        return self.entity_stats.get(entity_id, {})

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.NUM_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, config.NUM_FEATURES)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

class BaselineProfiler:
    def __init__(self):
        self.stat_profiler = StatisticalProfiler()
        self.autoencoder = Autoencoder()
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder.to(self.device)
        
    def fit(self, df, features_df, labels):
        self.stat_profiler.fit(df, features_df)
        
        # Train Autoencoder on normal data
        labels_arr = labels.values if hasattr(labels, 'values') else labels
        if labels_arr.dtype.kind in ('U', 'S', 'O'):  # string labels
            normal_idx = labels_arr == config.LABEL_NORMAL
        else:
            normal_idx = labels_arr == 0
            
        if not np.any(normal_idx):
            normal_idx = np.ones(len(labels_arr), dtype=bool)
            
        feat_values = features_df.values if hasattr(features_df, 'values') else features_df
        train_features = feat_values[normal_idx]
        train_features = self.scaler.fit_transform(train_features)
        
        dataset = TensorDataset(torch.FloatTensor(train_features))
        loader = DataLoader(dataset, batch_size=config.AE_BATCH_SIZE, shuffle=True)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=config.AE_LEARNING_RATE)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(config.AE_EPOCHS):
            for batch in loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                out = self.autoencoder(x)
                loss = criterion(out, x)
                loss.backward()
                optimizer.step()
                
    def score(self, df, features_df):
        scaled_features = self.scaler.transform(features_df.values)
        self.autoencoder.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(scaled_features).to(self.device)
            out = self.autoencoder(x)
            mse = torch.mean((out - x)**2, dim=1).cpu().numpy()
            
        # Normalize MSE to 0-1 range
        ae_scores = 1.0 - np.exp(-mse)
        
        stat_scores = []
        for i, row in enumerate(features_df.values):
            eid = df.iloc[i]['entity_id']
            stat_scores.append(self.stat_profiler.score(eid, row))
            
        stat_scores = np.array(stat_scores)
        
        # Combine
        combined = 0.5 * ae_scores + 0.5 * stat_scores
        return combined

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.autoencoder.state_dict(), path.with_suffix('.pt'))
        joblib.dump({
            'stat_profiler': self.stat_profiler,
            'scaler': self.scaler
        }, path.with_suffix('.joblib'))
        
    def load(self, path):
        path = Path(path)
        self.autoencoder.load_state_dict(torch.load(path.with_suffix('.pt'), map_location=self.device))
        data = joblib.load(path.with_suffix('.joblib'))
        self.stat_profiler = data['stat_profiler']
        self.scaler = data['scaler']
