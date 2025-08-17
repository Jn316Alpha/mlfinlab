import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mlfinlab.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from mlfinlab.util.utils import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import add_vertical_barrier, get_events, get_bins

# --- Configuration ---
# Input file paths
X_TRAIN_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\final_X_train.csv'
Y_TRAIN_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\final_y_train.csv'
SAMPLE_WEIGHT_TRAIN_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\final_sample_weight_train.csv'
X_TEST_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\final_X_test.csv'
Y_TEST_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\final_y_test.csv'
SAMPLE_WEIGHT_TEST_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\final_sample_weight_test.csv'

LABELS_RESULTS_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step2_triple_barrier_labels_results_8_15_25.csv'
DOLLAR_BARS_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step1_dollar_bars_es_results_8_15_25.csv'

if __name__ == "__main__":
    print("Loading final training and testing datasets...")
    X_train = pd.read_csv(X_TRAIN_PATH, index_col='date_time', parse_dates=True)
    y_train = pd.read_csv(Y_TRAIN_PATH, index_col='date_time', parse_dates=True).iloc[:, 0] # Assuming single column
    sample_weight_train = pd.read_csv(SAMPLE_WEIGHT_TRAIN_PATH, index_col='date_time', parse_dates=True).iloc[:, 0]

    X_test = pd.read_csv(X_TEST_PATH, index_col='date_time', parse_dates=True)
    y_test = pd.read_csv(Y_TEST_PATH, index_col='date_time', parse_dates=True).iloc[:, 0] # Assuming single column
    sample_weight_test = pd.read_csv(SAMPLE_WEIGHT_TEST_PATH, index_col='date_time', parse_dates=True).iloc[:, 0]

    # Load data for samples_info_sets and price_bars
    # Re-generate meta_labeled_events and extract t1 for samples_info_sets
    print("Re-generating meta_labeled_events for samples_info_sets...")
    # Load the original dollar bars data for feature generation
    data = pd.read_csv(DOLLAR_BARS_PATH, index_col='date_time', parse_dates=True)

    # Compute moving averages (as in test_sb_bagging.py)
    data['fast_mavg'] = data['close'].rolling(window=20, min_periods=20, center=False).mean()
    data['slow_mavg'] = data['close'].rolling(window=50, min_periods=50, center=False).mean()

    # Compute sides (as in test_sb_bagging.py)
    data['side'] = np.nan
    long_signals = data['fast_mavg'] >= data['slow_mavg']
    short_signals = data['fast_mavg'] < data['slow_mavg']
    data.loc[long_signals, 'side'] = 1
    data.loc[short_signals, 'side'] = -1
    data['side'] = data['side'].shift(1) # Remove Look ahead bias by lagging the signal

    # Generate events (as in test_sb_bagging.py)
    daily_vol = get_daily_vol(close=data['close'], lookback=50) * 0.5
    cusum_events = cusum_filter(data['close'], threshold=0.005)
    vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=data['close'], num_hours=2)
    meta_labeled_events = get_events(close=data['close'],
                                     t_events=cusum_events,
                                     pt_sl=[1, 4],
                                     target=daily_vol,
                                     min_ret=5e-5,
                                     num_threads=1, # Changed to 1 for simplicity in this script
                                     vertical_barrier_times=vertical_barriers,
                                     side_prediction=data['side'])
    meta_labeled_events.dropna(inplace=True)

    # Align all dataframes to the index of meta_labeled_events
    common_index = meta_labeled_events.index.intersection(X_train.index)
    X_train = X_train.loc[common_index]
    y_train = y_train.loc[common_index]
    sample_weight_train = sample_weight_train.loc[common_index]
    samples_info_sets = meta_labeled_events.loc[common_index, 't1']

    common_index_test = meta_labeled_events.index.intersection(X_test.index)
    X_test = X_test.loc[common_index_test]
    y_test = y_test.loc[common_index_test]
    sample_weight_test = sample_weight_test.loc[common_index_test]

    price_bars = data['close'] # Use the original close prices for price_bars
    print("meta_labeled_events re-generated and samples_info_sets extracted.")
    print(f"Aligned X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Aligned X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print("Initializing and training the Sequentially Bootstrapped Bagging Classifier...")
    # Define base estimator
    base_estimator = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                            class_weight='balanced_subsample', random_state=42)

    # Instantiate SequentiallyBootstrappedBaggingClassifier
    sb_classifier = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=base_estimator,
        samples_info_sets=samples_info_sets,
        price_bars=price_bars,
        n_estimators=100, # Number of base estimators
        max_samples=1.0,  # Use all samples for each base estimator
        max_features=1.0, # Use all features for each base estimator
        n_jobs=-1,        # Use all available CPU cores
        random_state=42   # For reproducibility
    )

    # Train the model
    sb_classifier.fit(X_train, y_train, sample_weight=sample_weight_train)
    print("Model training complete!")

    print("\n--- Model Evaluation on Test Set ---")
    # Make predictions
    y_pred = sb_classifier.predict(X_test)

    # Evaluate model
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)
