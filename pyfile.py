"""
EMAIL SPAM DETECTION - MODEL TRAINING SCRIPT
This script trains multiple machine learning models for spam detection
and saves them for later use.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Download stopwords
print("Downloading NLTK stopwords...")
nltk.download('stopwords')
print("✓ Download complete\n")
print("=" * 80)
print("STEP 1: LOADING DATASET")
print("=" * 80)

# Load the spam dataset
dataset_path = "spam.csv"  # Change this to your dataset path
print(f"Loading dataset from: {dataset_path}")

try:
    spam = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {spam.shape}")
    print(f"  - Columns: {list(spam.columns)}\n")
except FileNotFoundError:
    print(f"✗ Error: Dataset file '{dataset_path}' not found!")
    print("  Please ensure the spam.csv file is in the same directory.")
    exit(1)

# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================
print("=" * 80)
print("STEP 2: DATA EXPLORATION")
print("=" * 80)

print("\nFirst 5 rows of dataset:")
print(spam.head())

print("\nLast 5 rows of dataset:")
print(spam.tail())

print("\nChecking for null values:")
print(spam.isnull().sum())

print("\nDataset shape:", spam.shape)
print("STEP 3: DATA CLEANING")
print("=" * 80)

# Select only required columns
spam = spam[['v1', 'v2']]
spam.columns = ['label', 'message']
print("\nCleaned dataset (first 5 rows):")
print(spam.head())
print("\nClass distribution:")
class_counts = spam.groupby('label').size()
print(class_counts)
print(f"\nHAM: {class_counts['ham']} ({class_counts['ham']/len(spam)*100:.2f}%)")
print(f"SPAM: {class_counts['spam']} ({class_counts['spam']/len(spam)*100:.2f}%)")

print("\n" + "=" * 80)
print("STEP 4: DATA VISUALIZATION")
print("=" * 80)

print("\nGenerating class distribution plot...")
plt.figure(figsize=(10, 6))
spam['label'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Email Class Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved as 'class_distribution.png'\n")
plt.close()

# ============================================================================
# STEP 5: TEXT PREPROCESSING
# ============================================================================
print("=" * 80)
print("STEP 5: TEXT PREPROCESSING")
print("=" * 80)

print("\nPreprocessing steps:")
print("  1. Remove special characters")
print("  2. Convert to lowercase")
print("  3. Tokenization")
print("  4. Remove stopwords")
print("  5. Stemming")

# Initialize Porter Stemmer
ps = PorterStemmer()
corpus = []
print(f"\nProcessing {len(spam)} emails...")
for i in range(0, len(spam)):
    # Remove special characters
    review = re.sub('[^a-zA-Z]', ' ', spam['message'][i])

    # Convert to lowercase
    review = review.lower()
    
    # Tokenization
    review = review.split()
    
    # Remove stopwords and apply stemming
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    
    # Join back to string
    review = ' '.join(review)
    corpus.append(review)
    
    # Progress indicator
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i + 1}/{len(spam)} emails...")

print(f"✓ Preprocessing complete! Processed {len(corpus)} emails\n")

# Show examples
print("Preprocessing examples:")
print("-" * 80)
for i in range(3):
    print(f"\nOriginal  : {spam['message'][i][:100]}...")
    print(f"Processed : {corpus[i][:100]}...")
    print("-" * 80)

# ============================================================================
# STEP 6: FEATURE EXTRACTION (BAG OF WORDS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: FEATURE EXTRACTION")
print("=" * 80)

print("\nCreating Bag of Words model with CountVectorizer...")
print("  - Max features: 4000")

# Create CountVectorizer
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()

print(f"✓ Feature matrix created")
print(f"  - Shape: {X.shape}")
print(f"  - Number of features: {X.shape[1]}")

# Create labels
Y = pd.get_dummies(spam['label'])
Y = Y.iloc[:, 1].values  # 1 for spam, 0 for ham

print(f"✓ Labels encoded")
print(f"  - SPAM (1): {sum(Y)}")
print(f"  - HAM (0): {len(Y) - sum(Y)}\n")

print("=" * 80)
print("STEP 7: TRAIN-TEST SPLIT")
print("=" * 80)
test_size = 0.20
random_state = 42
print(f"\nSplitting dataset:")
print(f"  - Test size: {test_size * 100}%")
print(f"  - Random state: {random_state}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=random_state
)
print(f"\n✓ Dataset split complete")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")
print(f"  - Features: {X_train.shape[1]}\n")

print("=" * 80)
print("STEP 8: MODEL TRAINING")
print("=" * 80)

# Dictionary to store models and predictions
models = {}
predictions = {}
accuracies = {}

# Model 1: Random Forest Classifier
print("\n1. Training Random Forest Classifier...")
model1 = RandomForestClassifier(random_state=random_state, n_estimators=100)
model1.fit(X_train, Y_train)
pred1 = model1.predict(X_test)
models['RFC'] = model1
predictions['RFC'] = pred1
accuracies['RFC'] = accuracy_score(Y_test, pred1)
print(f"   ✓ Training complete - Accuracy: {accuracies['RFC']*100:.2f}%")

# Model 2: Decision Tree Classifier
print("\n2. Training Decision Tree Classifier...")
model2 = DecisionTreeClassifier(random_state=random_state)
model2.fit(X_train, Y_train)
pred2 = model2.predict(X_test)
models['DTC'] = model2
predictions['DTC'] = pred2
accuracies['DTC'] = accuracy_score(Y_test, pred2)
print(f"   ✓ Training complete - Accuracy: {accuracies['DTC']*100:.2f}%")

# Model 3: Multinomial Naïve Bayes
print("\n3. Training Multinomial Naïve Bayes...")
model3 = MultinomialNB()
model3.fit(X_train, Y_train)
pred3 = model3.predict(X_test)
models['MNB'] = model3
predictions['MNB'] = pred3
accuracies['MNB'] = accuracy_score(Y_test, pred3)
print(f"   ✓ Training complete - Accuracy: {accuracies['MNB']*100:.2f}%")

print("\n✓ All models trained successfully!\n")

# ============================================================================
# STEP 9: MODEL EVALUATION
# ============================================================================
print("=" * 80)
print("STEP 9: MODEL EVALUATION")
print("=" * 80)

# Create results comparison
print("\n" + "=" * 80)
print("MODEL ACCURACY COMPARISON")
print("=" * 80)

results_data = []
for model_name, acc in accuracies.items():
    model_full_names = {
        'RFC': 'Random Forest Classifier',
        'DTC': 'Decision Tree Classifier',
        'MNB': 'Multinomial Naïve Bayes'
    }
    results_data.append({
        'Model': model_full_names[model_name],
        'Accuracy': f"{acc*100:.2f}%"
    })
    print(f"{model_full_names[model_name]:<30} : {acc*100:.2f}%")

# Find best model
best_model_key = max(accuracies, key=accuracies.get)
best_model_names = {
    'RFC': 'Random Forest Classifier',
    'DTC': 'Decision Tree Classifier',
    'MNB': 'Multinomial Naïve Bayes'
}
print(f"\n🏆 Best Model: {best_model_names[best_model_key]} ({accuracies[best_model_key]*100:.2f}%)")

# Detailed evaluation for each model
for model_name, pred in predictions.items():
    print("\n" + "=" * 80)
    print(f"{best_model_names[model_name].upper()}")
    print("=" * 80)
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test, pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nInterpretation:")
    print(f"  - True Negatives (HAM correctly classified) : {cm[0][0]}")
    print(f"  - False Positives (HAM as SPAM)            : {cm[0][1]}")
    print(f"  - False Negatives (SPAM as HAM)            : {cm[1][0]}")
    print(f"  - True Positives (SPAM correctly classified): {cm[1][1]}")
    
    # Accuracy
    acc = accuracy_score(Y_test, pred)
    print(f"\nAccuracy: {acc*100:.2f}%")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(Y_test, pred, target_names=['HAM', 'SPAM']))
    
    # Save confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HAM', 'SPAM'],
                yticklabels=['HAM', 'SPAM'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(f'Confusion Matrix - {best_model_names[model_name]}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved as 'confusion_matrix_{model_name}.png'")
    plt.close()

# ============================================================================
# STEP 10: SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAVING MODELS")
print("=" * 80)

print("\nSaving trained models and vectorizer...")

# Save models
model_filenames = {
    'RFC': 'RFC.pkl',
    'DTC': 'DTC.pkl',
    'MNB': 'MNB.pkl'
}

for model_key, filename in model_filenames.items():
    with open(filename, 'wb') as f:
        pickle.dump(models[model_key], f)
    print(f"  ✓ Saved {best_model_names[model_key]:<30} -> {filename}")

# Save vectorizer
vectorizer_filename = 'vectorizer.pkl'
with open(vectorizer_filename, 'wb') as f:
    pickle.dump(cv, f)
print(f"  ✓ Saved CountVectorizer                    -> {vectorizer_filename}")

# Save preprocessing info
preprocessing_info = {
    'max_features': 4000,
    'test_size': test_size,
    'random_state': random_state,
    'accuracies': accuracies,
    'best_model': best_model_key
}

with open('preprocessing_info.pkl', 'wb') as f:
    pickle.dump(preprocessing_info, f)
print(f"  ✓ Saved Preprocessing Info                 -> preprocessing_info.pkl")

print("\n✓ All models saved successfully!\n")

# ============================================================================
# STEP 11: MODEL COMPARISON VISUALIZATION
# ============================================================================
print("=" * 80)
print("STEP 11: MODEL COMPARISON VISUALIZATION")
print("=" * 80)

print("\nGenerating model comparison chart...")

# Create comparison plot
fig, ax = plt.subplots(figsize=(12, 6))

model_names = [best_model_names[k] for k in accuracies.keys()]
acc_values = [acc * 100 for acc in accuracies.values()]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax.bar(model_names, acc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Customize plot
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_ylim([min(acc_values) - 5, 100])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, acc in zip(bars, acc_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Highlight best model
best_idx = acc_values.index(max(acc_values))
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comparison chart saved as 'model_comparison.png'\n")
plt.close()

# ============================================================================
# TRAINING COMPLETE
# ============================================================================
print("=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print("\n📊 Summary:")
print(f"  - Total emails processed: {len(spam)}")
print(f"  - Features extracted: {X.shape[1]}")
print(f"  - Models trained: {len(models)}")
print(f"  - Best model: {best_model_names[best_model_key]} ({accuracies[best_model_key]*100:.2f}%)")

print("\n💾 Saved files:")
print("  - RFC.pkl (Random Forest model)")
print("  - DTC.pkl (Decision Tree model)")
print("  - MNB.pkl (Multinomial Naïve Bayes model)")
print("  - vectorizer.pkl (CountVectorizer)")
print("  - preprocessing_info.pkl (Training metadata)")
print("  - class_distribution.png")
print("  - confusion_matrix_RFC.png")
print("  - confusion_matrix_DTC.png")
print("  - confusion_matrix_MNB.png")
print("  - model_comparison.png")

print("\n✅ All models are ready for deployment!")
print("=" * 80)