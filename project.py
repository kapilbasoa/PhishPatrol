#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier 
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib

#Loading the data
data0 = pd.read_csv('Phishing_Legitimate_full.csv')

#Checking the shape of the dataset
print(data0.shape)

#Listing the features of the dataset
print(data0.columns)

#Information about the dataset
print(data0.info())


#Plotting the data distribution
data0.hist(bins=50, figsize=(20,20))
plt.show()


# Selecting only numeric columns for correlation
numeric_data = data0.select_dtypes(include=[np.number])

# Plotting the correlation heatmap
plt.figure(figsize=(40,40))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.show()

# #Dropping the Domain column
data0= data0.drop(['id'], axis=1).copy()

# #checking the data for null or missing values
print(data0.isnull().sum())

# saving tghe feature orders
feature_order = ['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname',
                 'AtSymbol','TildeSymbol','NumUnderscore','NumPercent','NumQueryComponents',
                 'NumAmpersand','NumHash','NumNumericChars','NoHttps','RandomString','IpAddress',
                 'DomainInSubdomains','DomainInPaths','HttpsInHostname','HostnameLength','PathLength',
                 'QueryLength','DoubleSlashInPath','NumSensitiveWords','EmbeddedBrandName',
                 'PctExtHyperlinks','PctExtResourceUrls','ExtFavicon','InsecureForms','RelativeFormAction',
                 'ExtFormAction','AbnormalFormAction','PctNullSelfRedirectHyperlinks','FrequentDomainNameMismatch',
                 'FakeLinkInStatusBar','RightClickDisabled','PopUpWindow','SubmitInfoToEmail','IframeOrFrame',
                 'MissingTitle','ImagesOnlyInForm','SubdomainLevelRT','UrlLengthRT','PctExtResourceUrlsRT',
                 'AbnormalExtFormActionR','ExtMetaScriptLinkRT','PctExtNullSelfRedirectHyperlinksRT','CLASS_LABEL']

joblib.dump(feature_order, 'feature_order.pkl')  # Save the feature order

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data0.sample(frac=1).reset_index(drop=True)

# Separating features and target variable
y = data['CLASS_LABEL']  # Target column
X = data.drop('CLASS_LABEL', axis=1)


# Store column names
column_names = X.columns

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

#Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=12)


# Hyperparameter tuning for Decision Tree
param_grid_dt = {'max_depth': [5, 10, 15]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)
dt = DecisionTreeClassifier(**grid_search_dt.best_params_)
dt.fit(X_train, y_train)

# Hyperparameter tuning for Random Forest
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
rf = RandomForestClassifier(**grid_search_rf.best_params_)
rf.fit(X_train, y_train)

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
best_xgb.fit(X_train, y_train)

# Ensemble method with VotingClassifier
ensemble = VotingClassifier(estimators=[('dt', dt), ('rf', rf), ('xgb', best_xgb)], voting='soft')
ensemble.fit(X_train, y_train)

joblib.dump(ensemble, 'trained_ensemble_model.pkl')  # Save the ensemble model

# Predictions
y_test_dt = dt.predict(X_test)
y_test_rf = rf.predict(X_test)
y_test_xgb = best_xgb.predict(X_test)
y_test_ensemble = ensemble.predict(X_test)

# Accuracy
acc_train_dt = accuracy_score(y_train, dt.predict(X_train))
acc_test_dt = accuracy_score(y_test, y_test_dt)
acc_train_rf = accuracy_score(y_train, rf.predict(X_train))
acc_test_rf = accuracy_score(y_test, y_test_rf)
acc_train_xgb = accuracy_score(y_train, best_xgb.predict(X_train))
acc_test_xgb = accuracy_score(y_test, y_test_xgb)
acc_train_ensemble = accuracy_score(y_train, ensemble.predict(X_train))
acc_test_ensemble = accuracy_score(y_test, y_test_ensemble)

#Printing Accuracy
print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_dt))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_dt))
print("Random Forest: Accuracy on training Data: {:.3f}".format(acc_train_rf))
print("Random Forest: Accuracy on test Data: {:.3f}".format(acc_test_rf))
print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost: Accuracy on test Data: {:.3f}".format(acc_test_xgb))
print("Ensemble: Accuracy on training Data: {:.3f}".format(acc_train_ensemble))
print("Ensemble: Accuracy on test Data: {:.3f}".format(acc_test_ensemble))

# Classification Reports
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_test_dt))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_test_rf))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_test_xgb))
print("Ensemble Classification Report:")
print(classification_report(y_test, y_test_ensemble))

# Confusion Matrices
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_test_dt))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_rf))
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_test_xgb))
print("Ensemble Confusion Matrix:")
print(confusion_matrix(y_test, y_test_ensemble))

# Creating DataFrame to store results
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest','XGBoost', 'Ensemble'],
    'Train Accuracy': [acc_train_dt, acc_train_rf, acc_train_xgb, acc_train_ensemble],
    'Test Accuracy': [acc_test_dt, acc_test_rf,acc_test_xgb, acc_test_ensemble]
})

# Printing Results
print(results)

# Sorting Results by Test Accuracy
results_sorted = results.sort_values(by='Test Accuracy', ascending=False)
print(results_sorted)


def plot_roc_curve(y_test, predictions, model_name):
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.figure(figsize=(10, 8))
plot_roc_curve(y_test, dt.predict_proba(X_test)[:, 1], "Decision Tree")
plot_roc_curve(y_test, rf.predict_proba(X_test)[:, 1], "Random Forest")
plot_roc_curve(y_test, best_xgb.predict_proba(X_test)[:, 1], "XGBoost")
plot_roc_curve(y_test, ensemble.predict_proba(X_test)[:, 1], "Ensemble")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.show()


def plot_precision_recall_curve(y_test, predictions, model_name):
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    plt.plot(recall, precision, label=model_name)

plt.figure(figsize=(10, 8))
plot_precision_recall_curve(y_test, dt.predict_proba(X_test)[:, 1], "Decision Tree")
plot_precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1], "Random Forest")
plot_precision_recall_curve(y_test, best_xgb.predict_proba(X_test)[:, 1], "XGBoost")
plot_precision_recall_curve(y_test, ensemble.predict_proba(X_test)[:, 1], "Ensemble")

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.grid()
plt.show()


# top 10 important features for XGBoost
xgb_importances = best_xgb.feature_importances_
xgb_indices = np.argsort(xgb_importances)[::-1]

plt.figure(figsize=(10, 8))
plt.bar(range(10), xgb_importances[xgb_indices[:10]], align="center")
plt.xticks(range(10), X.columns[xgb_indices[:10]], rotation=45)
plt.title('Top 10 Feature Importances (XGBoost)')
plt.show()

