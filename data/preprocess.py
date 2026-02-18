"""
Preprocessing pipeline for REFIT dataset.
Assumes raw CSV files are in data/raw/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Appliance mapping (not strictly needed, but kept for reference)
APPLIANCES = {
    'kettle': ['kettle', 'Kettle'],
    'washing_machine': ['washing machine', 'Washing Machine', 'washer'],
    'dishwasher': ['dishwasher', 'Dishwasher'],
    'fridge': ['fridge', 'Fridge', 'refrigerator'],
    'freezer': ['freezer', 'Freezer'],
    'microwave': ['microwave', 'Microwave'],
    'television': ['television', 'TV', 'Television'],
    'monitor': ['monitor', 'Computer Monitor', 'screen'],
    'lighting': ['light', 'Lighting', 'lamp']
}

def load_household(filepath):
    """Load and resample household data to 1-minute intervals."""
    df = pd.read_csv(filepath, low_memory=False)
    
    # Convert time column (adjust based on actual REFIT format)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    
    # Resample to 1-minute
    df = df.resample('1T').mean()
    
    return df

def extract_features(window):
    """Extract features from a sliding window of aggregate power."""
    features = {}
    
    # Statistical features
    features['mean'] = window.mean()
    features['std'] = window.std()
    features['min'] = window.min()
    features['max'] = window.max()
    features['range'] = features['max'] - features['min']
    features['cv'] = features['std'] / (features['mean'] + 1e-8)  # Coefficient of variation
    
    # Change points
    features['diff_1min'] = window.iloc[-1] - window.iloc[-2] if len(window) > 1 else 0
    features['diff_5min'] = window.iloc[-1] - window.iloc[-6] if len(window) > 6 else 0
    features['diff_15min'] = window.iloc[-1] - window.iloc[-16] if len(window) > 16 else 0
    
    # Rolling statistics
    features['rolling_mean_5'] = window.rolling(5, min_periods=1).mean().iloc[-1]
    features['rolling_std_5'] = window.rolling(5, min_periods=1).std().iloc[-1]
    
    return features

def create_features_and_targets(df, aggregate_col='aggregate', appliance_cols=None):
    """
    Create feature matrix and target labels.
    aggregate_col : name of the column containing total house power.
    appliance_cols : list of column names for individual appliances.
    """
    if appliance_cols is None:
        # Assume all numeric columns except the aggregate are appliances
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        appliance_cols = [col for col in all_numeric if col != aggregate_col]
    
    features_list = []
    targets_list = []
    timestamps = []
    
    # Sliding window (60 minutes)
    window_size = 60
    
    for i in tqdm(range(window_size, len(df)), desc="Creating features"):
        # Window of aggregate power
        window = df[aggregate_col].iloc[i-window_size:i]
        
        # Extract features
        feat = extract_features(window)
        
        # Add temporal features
        feat['hour'] = df.index[i].hour
        feat['day_of_week'] = df.index[i].dayofweek
        feat['month'] = df.index[i].month
        feat['is_weekend'] = 1 if df.index[i].dayofweek >= 5 else 0
        
        features_list.append(feat)
        timestamps.append(df.index[i])
        
        # Determine which appliance is active (if any)
        active_appliances = []
        for app in appliance_cols:
            # Use a threshold of 15W to consider appliance ON
            if df[app].iloc[i] > 15:
                active_appliances.append(app)
        
        if len(active_appliances) == 0:
            targets_list.append('none')
        elif len(active_appliances) == 1:
            targets_list.append(active_appliances[0])
        else:
            # Multiple appliances active – choose the one with highest power
            powers = {app: df[app].iloc[i] for app in active_appliances}
            targets_list.append(max(powers, key=powers.get))
    
    features_df = pd.DataFrame(features_list, index=timestamps)
    targets_series = pd.Series(targets_list, index=timestamps)
    
    return features_df, targets_series

def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("REFIT Dataset Preprocessing")
    print("=" * 60)
    
    # Find raw data files
    raw_files = glob("data/raw/*.csv")
    if not raw_files:
        print("\n❌ No CSV files found in data/raw/")
        print("Please download the REFIT dataset first using data/download_data.py")
        return
    
    print(f"\nFound {len(raw_files)} household files")
    
    # Process each household
    all_features = []
    all_targets = []
    household_ids = []
    
    # Limit to a few households for quick testing; change as needed
    for i, filepath in enumerate(raw_files[:5]):  # Process up to 5 households
        print(f"\nProcessing household {i+1}/{min(5, len(raw_files))}: {os.path.basename(filepath)}")
        
        try:
            # Load data
            df = load_household(filepath)
            
            # Identify the aggregate column (first power column) and appliance columns
            power_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not power_cols:
                print(f"  ✗ No numeric columns found in {filepath}")
                continue
            
            aggregate_col = power_cols[0]   # Usually the first column is total house power
            appliance_cols = power_cols[1:] # The rest are individual appliances
            
            # Create features and targets
            features, targets = create_features_and_targets(
                df, aggregate_col, appliance_cols
            )
            
            all_features.append(features)
            all_targets.append(targets)
            household_ids.extend([i] * len(features))
            
            print(f"  ✓ Created {len(features)} samples")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Combine all households
    if all_features:
        combined_features = pd.concat(all_features)
        combined_targets = pd.concat(all_targets)
        combined_households = pd.Series(household_ids, index=combined_features.index)
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        combined_features.to_csv("data/processed/features.csv")
        combined_targets.to_csv("data/processed/targets.csv")
        combined_households.to_csv("data/processed/household_ids.csv")
        
        print("\n" + "=" * 60)
        print("✅ Preprocessing complete!")
        print("=" * 60)
        print(f"Total samples: {len(combined_features)}")
        print(f"Feature dimension: {combined_features.shape[1]}")
        print(f"Target classes: {combined_targets.nunique()}")
        print(f"Class distribution:")
        print(combined_targets.value_counts(normalize=True).head(10))
        print("\nFiles saved to: data/processed/")
    else:
        print("\n❌ No data processed successfully")

if __name__ == "__main__":
    main()
