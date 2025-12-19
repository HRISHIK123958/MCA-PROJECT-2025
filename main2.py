import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc)
import joblib
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory
if not os.path.exists('outputs'):
    os.makedirs('outputs')
    print(" Created 'outputs' directory")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

def load_dataset(filepath='new_dataset.csv'):
    """Load the banking firewall dataset"""
    print("\n" + "-"*80)
    print("STEP 1: LOADING DATASET")
    print("-"*80)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully from '{filepath}'")
        print(f"\n Dataset Overview:")
        print(f"   Total Records: {len(df)}")
        print(f"   Total Features: {len(df.columns)}")
        print(f"   Columns: {list(df.columns)}")
        
        print(f"\n Status Distribution:")
        status_counts = df['Status'].value_counts()
        for status, count in status_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {status}: {count} ({percentage:.1f}%)")
        
        print(f"\n First 5 Records:")
        print(df.head().to_string())
        
        return df
    
    except FileNotFoundError:
        print(f"❌ ERROR: File '{filepath}' not found!")
        print("Please ensure the dataset file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        return None

# ============================================================================
# STEP 2: PREPROCESS & FEATURE ENGINEERING
# ============================================================================

def preprocess_data(df):
    """Clean data and engineer features"""
    print("\n" + "-"*80)
    print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("-"*80)
    
    df = df.copy()
    
    # Extract hour from time
    df['Hour'] = df['Time'].apply(lambda t: int(str(t).split(':')[0]))
    print("✅ Extracted 'Hour' from 'Time'")
    
    # Feature 1: Is Foreign Location?
    df['IsForeign'] = (df['Location'] != 'India').astype(int)
    print("✅ Created 'IsForeign' feature (1 if not India, 0 otherwise)")
    
    # Feature 2: Is Night Time? (10 PM to 5 AM)
    df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
    print("✅ Created 'IsNight' feature (1 if between 10 PM - 5 AM)")
    
    # Feature 3: Is High Amount? (> 50000)
    df['IsHighAmount'] = (df['Amount'] > 50000).astype(int)
    print("✅ Created 'IsHighAmount' feature (1 if amount > 50000)")
    
    # Feature 4: Is Very High Amount? (> 80000)
    df['IsVeryHighAmount'] = (df['Amount'] > 80000).astype(int)
    print("✅ Created 'IsVeryHighAmount' feature (1 if amount > 80000)")
    
    # Feature 5: Device Type Encoding
    device_mapping = {'Laptop': 0, 'Desktop': 1, 'Mobile': 2, 'Tablet': 3}
    df['DeviceCode'] = df['Device'].map(device_mapping)
    print("✅ Encoded 'Device' as numeric (Laptop=0, Desktop=1, Mobile=2, Tablet=3)")
    
    # Feature 6: Amount Normalized (0-1 scale)
    df['AmountNormalized'] = (df['Amount'] - df['Amount'].min()) / \
                             (df['Amount'].max() - df['Amount'].min())
    print("✅ Normalized 'Amount' to 0-1 scale")
    
    # Create target label
    df['Label'] = (df['Status'] == 'Suspicious').astype(int)
    print("✅ Created 'Label' (1 for Suspicious, 0 for Normal)")
    
    # Select features for model
    feature_columns = ['IsForeign', 'IsNight', 'IsHighAmount', 'IsVeryHighAmount',
                      'DeviceCode', 'Hour', 'AmountNormalized']
    
    X = df[feature_columns]
    y = df['Label']
    
    print(f"\n Feature Summary:")
    print(f"   Total Features: {len(feature_columns)}")
    print(f"   Features: {feature_columns}")
    print(f"\n Target Distribution:")
    print(f"   Normal (0): {sum(y==0)} samples")
    print(f"   Suspicious (1): {sum(y==1)} samples")
    
    return X, y, df, feature_columns

# ============================================================================
# STEP 3: TRAIN  MODEL
# ============================================================================

def train_model(X, y, feature_columns):
    """Train Random Forest Classifier"""
    print("\n" + "-"*80)
    print("STEP 3: TRAINING MODEL")
    print("-"*80)
    
    # Split dataset (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"    Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Initialize Random Forest
    print("\n    Training Random Forest Classifier...")
    print("   Parameters:")
    print("   - Number of trees: 100")
    print("   - Max depth: 15")
    print("   - Min samples split: 5")
    print("   - Min samples leaf: 2")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Display results
    print("\n" + "="*80)
    print(" MODEL PERFORMANCE METRICS")
    print("="*80)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal  Suspicious")
    print(f"Actual Normal      {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Suspicious  {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    print(f"\n Interpretation:")
    print(f"   True Negatives (Correct Normal):      {cm[0][0]}")
    print(f"   False Positives (Normal as Suspicious): {cm[0][1]}")
    print(f"   False Negatives (Suspicious as Normal): {cm[1][0]}")
    print(f"   True Positives (Correct Suspicious):   {cm[1][1]}")
    
    # Classification report
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Normal', 'Suspicious'],
                               zero_division=0))
    
    # Feature importance
    print("\nFeature Importance (Top to Bottom):")
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance_df.iterrows():
        print(f"   {row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Save model
    model_path = 'outputs/random_forest_firewall_model.joblib'
    joblib.dump(rf_model, model_path)
    print(f"\n Model saved to: {model_path}")
    
    return rf_model, X_test, y_test, y_pred, y_pred_proba, feature_importance_df

# ============================================================================
# STEP 4: CREATE VISUALIZATIONS
# ============================================================================

def create_visualizations(y_test, y_pred, y_pred_proba, feature_importance_df):
    """Generate all visualization charts"""
    print("\n" + "-"*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("-"*80)
    
    # 1. Confusion Matrix Heatmap
    print("Creating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Suspicious'],
                yticklabels=['Normal', 'Suspicious'],
                annot_kws={"size": 16})
    plt.title('Confusion Matrix - Random Forest Firewall', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual Status', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Status', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/1_confusion_matrix.png")
    
    # 2. Feature Importance Chart
    print("Creating Feature Importance Chart...")
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance_df['Feature'], 
                    feature_importance_df['Importance'],
                    color='steelblue', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Feature Importance in Random Forest Model', 
              fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/2_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/2_feature_importance.png")
    
    # 3. ROC Curve
    print(" Creating ROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
             label=f'Random Forest (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Model Performance', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/3_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/3_roc_curve.png")
    
    # 4. Prediction Distribution
    print("Creating Prediction Distribution Chart...")
    pred_labels = ['Normal' if p == 0 else 'Suspicious' for p in y_pred]
    pred_counts = pd.Series(pred_labels).value_counts()
    
    plt.figure(figsize=(8, 6))
    colors = ['#90EE90', '#FF6B6B']
    wedges, texts, autotexts = plt.pie(pred_counts, labels=pred_counts.index, 
                                        autopct='%1.1f%%', colors=colors,
                                        startangle=90, textprops={'fontsize': 12})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Prediction Distribution (Test Set)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/4_prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/4_prediction_distribution.png")
    
    # 5. Model Metrics Comparison
    print(" Creating Metrics Comparison Chart...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'],
                   edgecolor='black', linewidth=1.5)
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/5_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/5_metrics_comparison.png")
    
    print("\n✅ All visualizations created successfully!")

# ============================================================================
# STEP 5: DEMO PREDICTIONS
# ============================================================================

def demo_predictions(rf_model, df, feature_columns):
    """Show sample predictions"""
    print("\n" + "-"*80)
    print("STEP 5: SAMPLE PREDICTIONS ")
    print("-"*80)
    
    # Select 10 random samples
    sample_df = df.sample(n=min(10, len(df)), random_state=42).copy()
    X_sample = sample_df[feature_columns]
    
    # Predict
    predictions = rf_model.predict(X_sample)
    probabilities = rf_model.predict_proba(X_sample)
    
    # Add predictions
    sample_df['AI_Prediction'] = ['Suspicious' if p == 1 else 'Normal' 
                                   for p in predictions]
    sample_df['Confidence'] = [f"{max(prob)*100:.1f}%" for prob in probabilities]
    sample_df['Match'] = ['✅' if sample_df.iloc[i]['Status'] == 
                          sample_df.iloc[i]['AI_Prediction'] else '❌' 
                          for i in range(len(sample_df))]
    
    # Display
    display_cols = ['Transaction_ID', 'User', 'Amount', 'Location', 'Time',
                   'Device', 'Status', 'AI_Prediction', 'Confidence', 'Match']
    
    print("\nSample Transactions with AI Predictions:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(sample_df[display_cols].to_string(index=False))
    
    # Accuracy on sample
    correct = sum(sample_df['Status'] == sample_df['AI_Prediction'])
    total = len(sample_df)
    print(f"\n Prediction Accuracy on Sample: {correct}/{total} " +
          f"({correct/total*100:.1f}%)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Step 1: Load data
    df = load_dataset(filepath='new_dataset.csv')
    if df is None:
        return
    
    # Step 2: Preprocess
    X, y, df, feature_columns = preprocess_data(df)
    
    # Step 3: Train model
    rf_model, X_test, y_test, y_pred, y_pred_proba, feature_importance_df = \
        train_model(X, y, feature_columns)
    
    # Step 4: Create visualizations
    create_visualizations(y_test, y_pred, y_pred_proba, feature_importance_df)
    
    # Step 5: Demo predictions
    demo_predictions(rf_model, df, feature_columns)
    
    # Final summary
    print("\n" + "="*80)
    print(" EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Final Model Accuracy: {accuracy*100:.2f}%")
    print(f"\n All outputs saved in 'outputs/' folder:")
    print("   - random_forest_firewall_model.joblib (trained model)")
    print("   - 1_confusion_matrix.png")
    print("   - 2_feature_importance.png")
    print("   - 3_roc_curve.png")
    print("   - 4_prediction_distribution.png")
    print("   - 5_metrics_comparison.png")
    print("="*80 + "\n")



if __name__ == "__main__":
    main()
