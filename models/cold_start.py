import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class ColdStartHandler:
    def __init__(self, statistical_profiler):
        self.statistical_profiler = statistical_profiler
        self.type_baselines = {}
        self.global_baseline = None
        
    def fit(self, features_df, entity_ids, entity_types):
        df_temp = features_df.copy()
        df_temp['entity_type'] = entity_types
        
        # Calculate type baselines
        for etype, group in df_temp.groupby('entity_type'):
            self.type_baselines[etype] = group.drop(columns=['entity_type']).mean().values
            
        self.global_baseline = df_temp.drop(columns=['entity_type']).mean().values
        
    def get_baseline(self, entity_id, entity_type, n_events):
        entity_stats = self.statistical_profiler.get_entity_stats(entity_id)
        entity_mean = entity_stats.get('mean')
        
        cohort_baseline = self.type_baselines.get(entity_type, self.global_baseline)
        
        if entity_mean is None or n_events < config.COLD_START_THRESHOLD:
            return cohort_baseline
            
        if n_events >= config.COLD_START_BLEND_EVENTS:
            return entity_mean
            
        # Blend
        alpha = (n_events - config.COLD_START_THRESHOLD) / (config.COLD_START_BLEND_EVENTS - config.COLD_START_THRESHOLD)
        return alpha * entity_mean + (1 - alpha) * cohort_baseline
        
    def score(self, entity_id, entity_type, n_events, feature_vector):
        baseline = self.get_baseline(entity_id, entity_type, n_events)
        dist = np.linalg.norm(feature_vector - baseline)
        return 1.0 - np.exp(-dist)
