import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys
import traceback

# --- Configuration ---
DOLLAR_BARS_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step1_dollar_bars_es_results_8_15_25.csv'
LABELS_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step2_triple_barrier_labels_results_8_15_25.csv'
FD_FEATURES_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step4_fd_features.csv'
COT_FEATURES_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step4_cot_features.csv'
PURGED_EMBARGOED_INDICES_PATH = r'C:\Users\pfa34\MLFinlab\mlfinlab\step5_purged_embargoed_indices.csv'

SHORT_MA = 10
LONG_MA = 30

# --- Helper Functions ---
def calculate_primary_side(df_close, short_window, long_window):
    df_close = pd.DataFrame(df_close)
    short_mavg = df_close.rolling(window=short_window, min_periods=1).mean()
    long_mavg = df_close.rolling(window=long_window, min_periods=1).mean()
    side = pd.Series(0, index=df_close.index)
    side[short_mavg.iloc[:,0] > long_mavg.iloc[:,0]] = 1
    side[short_mavg.iloc[:,0] < long_mavg.iloc[:,0]] = -1
    return side

def get_meta_labels(labels_df, side_signals):
    merged_data = labels_df.copy()
    merged_data = merged_data.merge(side_signals.rename('side'), left_index=True, right_index=True, how='left')
    merged_data.dropna(subset=['side'], inplace=True)
    merged_data['prod_ret'] = merged_data['ret'] * merged_data['side']
    merged_data['meta_bin'] = (merged_data['prod_ret'] > 0).astype(int)
    return merged_data['meta_bin']

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Meta-Labeling Process ---")
    try:
        # 1. Load Data
        print("Loading data...")
        dollar_bars_df = pd.read_csv(DOLLAR_BARS_PATH, index_col='date_time', parse_dates=True)
        labels_df = pd.read_csv(LABELS_PATH, index_col='Unnamed: 0', parse_dates=True)
        labels_df.index.name = 'date_time'
        fd_features_df = pd.read_csv(FD_FEATURES_PATH, index_col='date_time', parse_dates=True)
        cot_features_df = pd.read_csv(COT_FEATURES_PATH, index_col='date_time', parse_dates=True)
        purged_embargoed_indices = pd.read_csv(PURGED_EMBARGOED_INDICES_PATH, index_col='date_time', parse_dates=True)
        print("Data loaded successfully.")

        # 2. Define Primary Strategy
        print(f"Generating primary strategy 'side' signals using {SHORT_MA}-period and {LONG_MA}-period SMAs...")
        primary_side_signals = calculate_primary_side(dollar_bars_df['close'], SHORT_MA, LONG_MA)
        primary_side_signals = primary_side_signals.reindex(labels_df.index).dropna()
        print("Primary strategy 'side' signals generated.")

        # 3. Generate Meta-Labels
        print("Generating meta-labels...")
        if 'ret' not in labels_df.columns:
            print("Error: 'ret' column not found in labels_df. Cannot generate meta-labels.")
            sys.exit()
        meta_labels = get_meta_labels(labels_df[['ret']], primary_side_signals)
        print(f"Meta-labels generated. Class distribution:\n{meta_labels.value_counts(normalize=True)}")

        # 4. Combine Features
        print("Combining features...")
        features_df = fd_features_df.join(cot_features_df, how='inner')
        print(f"Combined features. Shape: {features_df.shape}")

        # Align features and labels
        print("Aligning features and labels...")
        aligned_data = features_df.join(meta_labels, how='inner')
        X = aligned_data.drop(columns=['meta_bin'])
        y = aligned_data['meta_bin']
        print(f"Aligned data. X shape: {X.shape}, y shape: {y.shape}")

        # 5. Train/Test Split
        print("Splitting data into training and testing sets...")
        train_indices = purged_embargoed_indices[purged_embargoed_indices['is_train']].index
        test_indices = purged_embargoed_indices[purged_embargoed_indices['is_test']].index
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        print(f"Data split. X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

        # 6. Handle NaN values
        print("Handling NaN values...")
        X_train.dropna(inplace=True)
        y_train = y_train.loc[X_train.index]
        X_test.dropna(inplace=True)
        y_test = y_test.loc[X_test.index]
        print("Rows with NaN values dropped.")

        # 7. Train Model
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
        model.fit(X_train, y_train)
        print("Model training complete.")

        # 8. Evaluate Model
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        print("--- Classification Report ---")
        print(report)
        print("--- Confusion Matrix ---")
        print(matrix)

        # 9. Save Outputs
        print("Saving outputs...")
        joblib.dump(model, 'final_model.joblib')
        X_train.to_csv('final_X_train.csv')
        X_test.to_csv('final_X_test.csv')
        y_train.to_csv('final_y_train.csv')
        y_test.to_csv('final_y_test.csv')
        with open('S7_meta_labeling_results.txt', 'w') as f:
            f.write("--- Classification Report ---\n")
            f.write(report)
            f.write("\n--- Confusion Matrix ---\n")
            f.write(str(matrix))
        print("Outputs saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

    print("--- Meta-Labeling Process Finished ---")
