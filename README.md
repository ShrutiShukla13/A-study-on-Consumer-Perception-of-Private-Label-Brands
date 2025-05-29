# A-study-on-Consumer-Perception-of-Private-Label-Brands
Analyzed survey data (n=109) to study consumer perception of Private Label Brands using Python and ML models. SVM achieved 96% accuracy, revealing quality and brand familiarity as key drivers. Young, budget-conscious buyers showed strong preference, aiding targeted retail strategies.

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:\\Users\\ABHISHEK\\OneDrive\\Attachments\\Desktop\\All Projects\\orgdata.csv"
df=pd.read_csv(file_path)
# Quick info about data
print("\n--- BASIC INFORMATION ---")
print(df.info())

# Checking missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# First few rows
print("\n--- SAMPLE DATA ---")
print(df.head())

# Renaming columns to simpler names for easier handling
df.columns = [col.strip().replace(':', '').replace('?', '').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_') for col in df.columns]

# Let's now visualize!

# Plot 1: Age Distribution
plt.figure(figsize=(8,5))
sns.countplot(y=df['Age'])
plt.title('Age Group Distribution')
plt.xlabel('Count')
plt.ylabel('Age Group')
plt.show()

# Plot 2: Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['Gender'])
plt.title('Gender Distribution')
plt.show()

# Plot 3: Household Income Distribution
plt.figure(figsize=(10,6))
sns.countplot(y=df['Household_Income'])
plt.title('Household Income Distribution')
plt.show()

# Plot 4: Familiarity with Private Brands
plt.figure(figsize=(10,5))
sns.countplot(y=df['How_familiar_are_you_with_private_brands_store_brands'])
plt.title('Familiarity with Private Brands')
plt.show()

# Plot 5: Satisfaction Rating (assuming 1-5 scale)
plt.figure(figsize=(7,4))
sns.countplot(x=df['How_satisfied_are_you_with_private_brands_you_have_purchased'])
plt.title('Satisfaction with Private Brands')
plt.show()

# Plot 6: Recommendation Likelihood
plt.figure(figsize=(7,4))
sns.countplot(x=df['How_likely_are_you_to_recommend_private_brands_to_others'])
plt.title('Likelihood to Recommend Private Brands')
plt.show()

# 🔥 Correlation Heatmap (Numerical Features Only)
numeric_df = df.select_dtypes(include=['number'])

if not numeric_df.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', fmt=".2f", square=True, linewidths=0.5)
    plt.title("🔥 Clean Correlation Heatmap (Numerical Features Only)")
    plt.tight_layout()
    plt.show()
else:
    print("⚠ No numeric columns found for correlation heatmap.")
# Additional Tip: Summary insights
print("BASIC SUMMARY INSIGHTS")
print("Most common Age group:", df['Age'].mode()[0])
print("Most common Gender:", df['Gender'].mode()[0])
print("Most common Income Group:", df['Household_Income'].mode()[0])
print("Most common Familiarity Level:", df['How_familiar_are_you_with_private_brands_store_brands'].mode()[0])
#Key Insights Summary
#Young, male, low-income respondents dominate the sample.
###############################################################################################

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "C:\\Users\\ABHISHEK\\OneDrive\\Attachments\\Desktop\\All Projects\\orgdata.csv"
df = pd.read_csv(file_path)

# Define multiple target columns
target_columns = [
    'What_is_your_overall_perception_of_private_brands_compared_to_national_brands',
    'How_likely_are_you_to_recommend_private_brands_to_others',
    'What_is_your_overall_perception_of_private_brands_compared_to_national_brands'
]

# Store all selected features across target variables
all_selected_features = set()

for target_column in target_columns:
    # Drop rows with missing values in the target column
    df_target = df.dropna(subset=[target_column])

    # Split into features and target
    X = df_target.drop(columns=target_columns)  # drop all target columns from features
    y = df_target[target_column]

    # Encode target if needed
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Apply chi-square feature selection
    selector = SelectKBest(score_func=chi2, k=10)
    X_new = selector.fit_transform(X, y_encoded)

    # Add selected features to the set
    selected = X.columns[selector.get_support()].tolist()
    all_selected_features.update(selected)

# Convert set to sorted list
final_features = sorted(all_selected_features)

# Final dataframe with selected features and all target columns
df_selected = df[final_features + target_columns]

# Save to CSV
save_path = "C:\\Users\\ABHISHEK\\Downloads\\Featured.csv"
df_selected.to_csv(save_path, index=False)

print("✅ Selected features across all target columns:")
print(final_features)
print(f"\n✅ Final dataset saved at:\n{save_path}")

####### Done with model building
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# 2. Load Data
file_path = "C:\\Users\\ABHISHEK\\OneDrive\\Attachments\\Desktop\\All Projects\\encoded_orgdata_1.csv"
df = pd.read_csv(file_path)

# 3. Set Multiple Target Columns
target_columns = [
    'What_is_your_overall_perception_of_private_brands_compared_to_national_brands',
    'How_likely_are_you_to_recommend_private_brands_to_others',
    'How_often_do_you_purchase_private_brands'
]
# 4. Function to Evaluate Models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cmap='Blues'):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap=cmap)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# 5. Run Models for Each Target
for target_column in target_columns:
    print(f"\n\n{'='*60}\n📌 Evaluating Models for Target: {target_column}\n{'='*60}")
    
    X = df.drop(target_columns, axis=1)
    y = df[target_column]
    
    # Encode target if categorical
    if y.dtype == 'object' or y.nunique() > 2:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    evaluate_model(LogisticRegression(max_iter=1000), X_train, y_train, X_test, y_test, "Logistic Regression")

    # Decision Tree
    model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    evaluate_model(model_dt, X_train, y_train, X_test, y_test, "Decision Tree", cmap='Greens')

    plt.figure(figsize=(20,10))
    plot_tree(model_dt, feature_names=X.columns.astype(str), class_names=np.unique(y_train).astype(str), filled=True)
    plt.title("Decision Tree Structure")
    plt.show()

    # Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    evaluate_model(model_rf, X_train, y_train, X_test, y_test, "Random Forest", cmap='Purples')

    # KNN
    evaluate_model(KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test, "K-Nearest Neighbors")

    # SVM
    evaluate_model(SVC(kernel='linear', probability=True, random_state=42), X_train, y_train, X_test, y_test, "Support Vector Machine", cmap='coolwarm')

    # Naive Bayes
    evaluate_model(GaussianNB(), X_train, y_train, X_test, y_test, "Naive Bayes")

    # Gradient Boosting
    model_gbc = GradientBoostingClassifier(random_state=42)
    evaluate_model(model_gbc, X_train, y_train, X_test, y_test, "Gradient Boosting", cmap='cividis')

    # Optional: AUC for GBC
    try:
        gbc_probs = model_gbc.predict_proba(X_test)
        gbc_auc = roc_auc_score(y_test, gbc_probs, multi_class='ovr')
        print(f"Gradient Boosting AUC Score (OvR): {gbc_auc:.2f}")
    except:
        pass

    # Optional: OOB error for Random Forest
    rf_oob_model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    rf_oob_model.fit(X_train, y_train)
    oob_err = 1 - rf_oob_model.oob_score_
    print(f"Random Forest OOB Error: {oob_err:.4f}")
