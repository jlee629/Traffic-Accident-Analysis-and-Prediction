# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:11:59 2023

@author: 8778t

Group 6:
Alejandro Akifarry
Jungyu Lee
Minyoung Seol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import webbrowser
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
# ======================= 1. DATA EXPLORATION =================================
# read csv file
df_ksi = pd.read_csv("C:/Users/Public/4th/COMP247/W13/KSI.csv")

# column names, types, and info
df_ksi.head()
df_ksi.dtypes
df_ksi.info()

# statistical assessments including means, averages, correlations
df_ksi.describe()
df_ksi.corr()

# missing data evaluation
df_ksi.isnull().sum()
# print the null percentage of each column
print(df_ksi.isna().sum() / len(df_ksi) * 100)

# graph and visualization
"""
Due to the presence of numerous null values,
it is hard to produce meaningful graphs and visualizations.
For this reason, we will present them in the Data Modeling part,
after handling some of the missing data,
"""
# ===================== 2. DATA MODELLING =====================================
# data transformations
# reformat date and time for processing
df_ksi['DATE'] = pd.to_datetime(df_ksi['DATE'])

# adds relevant columns based on the DATE and TIME column
df_ksi['MONTH'] = df_ksi['DATE'].dt.month
df_ksi['DAY'] = df_ksi['DATE'].dt.day
df_ksi['HOUR'] = df_ksi['TIME'] // 100
df_ksi['MINUTES'] = df_ksi['TIME'] % 100
df_ksi['WEEKDAY'] = df_ksi['DATE'].dt.weekday

# drop both DATE and TIME after extracting info above
df_ksi = df_ksi.drop(['DATE', 'TIME'], axis=1)

# drop unique identifiers
df_ksi = df_ksi.drop(['X','Y','INDEX_', 'ObjectId', 'ACCNUM', 'FATAL_NO'],axis=1)

print(df_ksi.isna().sum()/len(df_ksi)*100)
# reformat data first before filling missing values
# these columns seems to only have 'Yes' and nan, so assume nan is 'No'
for attribute in ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']:
    df_ksi[attribute] = df_ksi[attribute].map({'Yes': 1, '': 0}).fillna(0).astype(int)

# reformat ACCLASS and make it into the target column FATAL
# 1 for fatal, 0 for non-fatal. fill missing values with 0
df_ksi['FATAL'] = df_ksi['ACCLASS'].map({'Non-Fatal Injury': 0, 'Property Damage Only': 0, 'Fatal': 1}).fillna(0)
df_ksi = df_ksi.drop(['ACCLASS'], axis=1)

# reformat DISTRICT class, combine 'Toronto and East York' and
# 'Toronto East York'
df_ksi['DISTRICT'] = df_ksi['DISTRICT'].replace({"Toronto East York": "Toronto and East York"})

# drop columns with missing data over 80%
for label, content in df_ksi.items():
    missing_count = df_ksi[label].isna().sum()
    missing_percentage = missing_count / len(df_ksi[label]) * 100
    if (missing_percentage > 80.0):
        df_ksi = df_ksi.drop([label], axis=1)

# drop location columns (except for LATITUDE, LONGITUDE, DISTRICT for visualizations)
df_ksi = df_ksi.drop(['ACCLOC', 'WARDNUM', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'STREET1', 'STREET2', 'ROAD_CLASS', 'LOCCOORD'],axis=1)

# drop unrelated columns due to missing data, not worth for prediction, too many & specific values
df_ksi = df_ksi.drop(['INJURY', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'DIVISION', 'TRAFFCTL', 'IMPACTYPE', 'INVTYPE'],axis=1)

# put INVAGE into several classes
# 0-29 (class1), 30-59 (class2), 60 and over (class3), unknown (class4)
class1names = [
    '0 to 4',
    '5 to 9',
    '10 to 14',
    '15 to 19',
    '20 to 24',
    '25 to 29'
]
class2names = [
    '30 to 34',
    '35 to 39',
    '40 to 44',
    '45 to 49',
    '50 to 54',
    '55 to 59'
]
class3names = [
    '60 to 64',
    '65 to 69',
    '70 to 74',
    '75 to 79',
    '80 to 84',
    '85 to 89',
    '90 to 94',
    'over 95'
]
new_invage = list()
for index, value in df_ksi['INVAGE'].items():
    new_invage_val = 'unknown'
    if value in class1names:
        new_invage_val = '0 to 29'
    elif value in class2names:
        new_invage_val = '30 to 59'
    elif value in class3names:
        new_invage_val = '60 and above'
    new_invage.append(new_invage_val)
# replace INVAGE with the new list
df_ksi['INVAGE'] = new_invage

## heatmap for latitude and longitude
df_ksi_map = df_ksi[['LATITUDE', 'LONGITUDE', 'FATAL']]
lat_Toronto = df_ksi_map.describe().at['mean', 'LATITUDE']
lng_Toronto = df_ksi_map.describe().at['mean', 'LONGITUDE']
Toronto_location = [lat_Toronto, lng_Toronto]

Toronto_map = folium.Map(location=Toronto_location, zoom_start=11)

heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['FATAL']] for index, row in df_ksi.iterrows()]

HeatMap(heat_data).add_to(Toronto_map)

file_path = "toronto_heatmap.html"
Toronto_map.save(file_path)

webbrowser.open(file_path)

# Year-Month Distribution of Fatal Accidents in Ontario
data = df_ksi.groupby(['YEAR', 'MONTH'])['FATAL'].sum().unstack(fill_value=0).astype(int)

plt.figure(figsize=(12, 6))
sns.heatmap(data, annot=True, fmt="d", cmap="Reds")
plt.title('Year-Month Distribution of Fatal Accidents in Toronto', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()

## Driving Condition
ksi_pivot_cause = df_ksi.pivot_table(
    index='YEAR',
    values=['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'],
    aggfunc='sum',
    margins=True,
)

colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

plt.figure(figsize=(10, 6))
ksi_pivot_cause.iloc[-1].plot(kind='bar', color=colors, edgecolor='black')

plt.xlabel('Driving Condition', fontsize=14)
plt.ylabel('Accidents Count', fontsize=14)
plt.title('Driving Condition vs Accidents in Toronto', fontsize=16)
plt.xticks(rotation=0)
plt.show()

## Type of Vehicle Involved
vehicle_involved = df_ksi.pivot_table(index='YEAR', values=['AUTOMOBILE', 'CYCLIST', 'EMERG_VEH', 'MOTORCYCLE', 'TRUCK'], aggfunc='sum')
vehicle_involved.plot(kind='bar', figsize=(10, 8), title="Type of Vehicle Involved", grid=True)
plt.ylabel('Vehicles')
plt.show()

# Type of Victims Involved
ksi_pivot_cpp = df_ksi.pivot_table(
    index='YEAR',
    values=['CYCLIST', 'PEDESTRIAN', 'PASSENGER'],
    aggfunc='sum',
    margins=True,
)

plt.figure(figsize=(8, 8))
ksi_pivot_cpp.iloc[-1].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.xlabel('Victims')
plt.ylabel('Number of Accidents')
plt.title('Type of Victims Involved', fontsize=16)
plt.xticks(rotation=0)
plt.show()

# a column drop for feature selection (not worth for prediction)
df_ksi = df_ksi.drop(['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTES', 'WEEKDAY', 'LATITUDE', 'LONGITUDE'], axis=1)

# Pipeline for numeric features
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
num_pipeline = Pipeline([
    ('imp_median', imp_median)
])

# Pipeline for categorical features
imp_common = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
onehot = OneHotEncoder(handle_unknown='ignore')
cat_pipeline = Pipeline([
    ('imp_common', imp_common),
    ('onehot', onehot)
])

# do oversampling to balance data
majority_samples = df_ksi[df_ksi['FATAL'] == 0] # get 0 class
minority_samples = df_ksi[df_ksi['FATAL'] == 1] # get 1 class

# oversampling minority class
oversampled_minority = np.random.choice(minority_samples.index,
                                        size=majority_samples.shape[0],
                                        replace=True)
oversampled_minority = minority_samples.loc[oversampled_minority]
# concat the majority and oversampled minority to make a balanced dataset
df_oversampled = pd.concat([majority_samples, oversampled_minority])
df_oversampled = df_oversampled.sample(frac=1, random_state=80)

# now class 0 and 1 in FATAL should have equal counts
df_oversampled['FATAL'].value_counts()

# separate features and target
X = df_oversampled.iloc[:, :-1]
y = df_oversampled.iloc[:, -1]

# save numerical and categorical feature names
# to use it in ColumnTransformer to label which feature is which
num_features = X.select_dtypes(exclude=['object']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()
print(num_features)
print(cat_features)

# define column transformer to preprocess those features
preprocessor = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, num_features),
    ('cat_pipeline', cat_pipeline, cat_features)
])

# Split train and test 80-20 with StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=80)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# use preprocessor to transform train and test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# convert the preprocessed data to a DataFrame
transformed_num_features = num_features
transformed_cat_features = \
    preprocessor.named_transformers_['cat_pipeline']['onehot']\
        .get_feature_names_out(cat_features).tolist()
transformed_col_names = transformed_num_features + transformed_cat_features
X_train_preprocessed = pd.DataFrame(X_train_preprocessed,\
                                    columns=transformed_col_names)

# ===================== 3. Predictive Model Building =====================================
cv=5
"""
logistic Regression
"""

# Initializing the Logistic Regression model
lrclf = LogisticRegression(random_state=42)

# Creating a pipeline combining the preprocessor and the classifier
pipeline_lrclf = Pipeline([
    ('col_transformer', preprocessor),
    ('lrclf', lrclf)
])

# Fitting the pipeline on training data
pipeline_lrclf.fit(X_train, y_train)

# Setting the hyperparameter grid for tuning
param_grid_lrclf = {
    'lrclf__solver': ['lbfgs', 'saga'],
    'lrclf__max_iter': [400, 500, 600, 700],
    'lrclf__tol': [0.001, 0.01, 0.1]
}

# Grid search for hyperparameter tuning
grid_search_lrclf = GridSearchCV(
    estimator=pipeline_lrclf,
    param_grid=param_grid_lrclf,
    cv=cv,
    error_score='raise',
    scoring='accuracy')

# Fitting the grid search model
grid_search_lrclf.fit(X_train, y_train)

best_model_lrclf = grid_search_lrclf.best_estimator_

# Printing the results
print('Best parameters of logistic regression classifier:')
print(grid_search_lrclf.best_params_)

"""
Decision Trees
"""
# Initializing the Decision Tree classifier
dtclf = DecisionTreeClassifier(random_state=42)

# Creating a pipeline combining the preprocessor and the classifier
pipeline_dtclf = Pipeline([
    ('col_transformer', preprocessor),
    ('dtclf', dtclf)
])

# Fitting the pipeline on training data
pipeline_dtclf.fit(X_train, y_train)

# Setting the parameter distributions for random search
param_dist_dtclf = {
    'dtclf__criterion': ['gini', 'entropy'],
    'dtclf__max_depth': [None, 10, 20, 30, 40, 50],
    'dtclf__min_samples_split': [2, 5, 10],
    'dtclf__min_samples_leaf': [1, 2, 4]
}

# Random search for hyperparameter tuning
random_search_dtclf = RandomizedSearchCV(estimator=pipeline_dtclf,
                                         param_distributions=param_dist_dtclf,
                                         n_iter=50,
                                         cv=cv,
                                         error_score='raise',
                                         scoring='accuracy',
                                         random_state=42)

# Fitting the random search model
random_search_dtclf.fit(X_train, y_train)

# Predicting using the best model
best_model_dtclf = random_search_dtclf.best_estimator_

# Printing the best parameters
print('Best parameters of Decision Tree:')
print(random_search_dtclf.best_params_)

"""
SVC
"""
svc = SVC(random_state=42)
pipeline_svc = Pipeline([
    ('col_transformer', preprocessor),
    ('svc', svc)
])

pipeline_svc.fit(X_train, y_train)

param_grid_svc = {
    'svc__kernel': ['rbf'],
    'svc__C': [1, 10],
    'svc__gamma': ['scale']
}

# Grid search for hyperparameter tuning
grid_search_svc = GridSearchCV(estimator=pipeline_svc,
                               param_grid=param_grid_svc,
                               cv=cv,
                               error_score='raise',
                               scoring='accuracy')

# Fitting the grid search model
grid_search_svc.fit(X_train, y_train)

# Predicting using the best model
best_model_svc = grid_search_svc.best_estimator_

# Printing the best parameters
print('Best parameters of SVM:')
print(grid_search_svc.best_params_)

"""
Random Forests
"""
# Initializing the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)

# Creating a pipeline combining the preprocessor and the classifier
pipeline_rf = Pipeline([
    ('col_transformer', preprocessor),
    ('rf', rf_clf)
])

# Parameter grid for Random Forest
param_dist_rf = {
    'rf__n_estimators': [10, 50, 100, 150, 200],
    'rf__max_depth': [None, 10, 20, 30, 40, 50],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__bootstrap': [True, False]
}

# Random search for hyperparameter tuning
random_search_rf = RandomizedSearchCV(estimator=pipeline_rf,
                                      param_distributions=param_dist_rf,
                                      cv=cv,
                                      error_score='raise',
                                      scoring='accuracy',
                                      random_state=42)

# Fitting the random search model
random_search_rf.fit(X_train, y_train)

# Predicting using the best model
best_model_rf = random_search_rf.best_estimator_

# Printing the best parameters
print('Best parameters of Random Forest:')
print(random_search_rf.best_params_)

"""
Neural networks
"""
# Initializing the MLP model with a random state for reproducibility
mlp = MLPClassifier(random_state=42, max_iter=5000) 

# Creating a pipeline combining the preprocessor and the MLP classifier
pipeline_mlp = Pipeline([
    ('col_transformer', preprocessor),
    ('mlp', mlp)
])

# Simplified hyperparameters to be tuned
param_dist_mlp = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam'],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__early_stopping': [True],
    'mlp__validation_fraction': [0.1],
    'mlp__n_iter_no_change': [10]
}

# Random search for hyperparameter tuning
random_search_mlp = RandomizedSearchCV(estimator=pipeline_mlp,
                                       param_distributions=param_dist_mlp,
                                       n_iter=10,  # Number of parameter settings sampled
                                       cv=cv,
                                       error_score='raise',
                                       scoring='accuracy',
                                       random_state=42)

# Fitting the random search model
random_search_mlp.fit(X_train, y_train)

# Predicting using the best model
best_model_mlp = random_search_mlp.best_estimator_

# Printing the best parameters
print('Best parameters of MLP:')
print(random_search_mlp.best_params_)


# ===================== 4. Predictive Model Building =====================================
# Create a list of models and their names for easier processing
models = [
    (best_model_lrclf, "Logistic Regression"),
    (best_model_dtclf, "Decision Tree"),
    (best_model_svc, "SVM"),
    (best_model_rf, "Random Forest"),
    (best_model_mlp, "MLP Neural Network")
]

# Initialize an empty list to store ROC curve values
best_auc = 0
best_models = []

roc_data = []

for model, name in models:
    y_pred = model.predict(X_test)
    
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("-" * 60)
    
    # Calculate ROC curve and store values only if predict_proba is available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_value = auc(fpr, tpr)
        roc_data.append((fpr, tpr, roc_auc_value, name))
        if roc_auc_value > best_auc:
            best_auc = roc_auc_value
            best_models = [name]
        elif roc_auc_value == best_auc:
            best_models.append(name)
            
# Plotting all ROC curves in a single figure
plt.figure(figsize=(10,8))
for fpr, tpr, roc_auc_value, name in roc_data:
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_value:.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

if len(best_models) == 1:
    print(f"The best-performing model is {best_models[0]} with an AUC of {best_auc:.2f}")
else:
    best_models_str = ', '.join(best_models)
    print(f"The models with the highest AUC of {best_auc:.2f} are: {best_models_str}")

with open('C:/Users/Public/4th/COMP247/W13/COMP247001_Project#1_Group6/best_model_rf.pkl', 'wb') as model_file:
    pickle.dump(best_model_rf, model_file)

