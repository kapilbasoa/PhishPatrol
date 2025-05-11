# PhishPatrol
Phishing websites are malicious platforms that are specifically designed to deceive users and steal sensitive information such as login credentials, credit card numbers, and other personal data. These websites are often designed to closely resemble legitimate websites—such as banking, e-commerce, or social media sites—by mimicking their visual appearance, structure, and even domain names. However, they typically have subtle differences that can indicate malicious intent, such as unusual URL patterns, misspelled domain names, suspicious subdomains, or insecure connections.
Phishing attacks are a significant threat in today’s digital world, as they exploit the trust that users place in well-known websites. Once users fall victim to a phishing attack, their personal data may be misused for identity theft, financial fraud, or other malicious activities. Traditional methods of detecting phishing websites, such as manually inspecting URLs or using blacklists, are often ineffective due to the constantly evolving nature of phishing tactics. Therefore, automated systems that can quickly and accurately detect phishing websites are crucial in preventing these attacks and safeguarding users.

The goal of this project is to leverage machine learning techniques to build a classifier capable of analyzing URLs and identifying potentially malicious websites based on various features. These features include the length of the URL, the number of subdomains, the presence of certain symbols (e.g., dashes, underscores), and domain-specific patterns that are characteristic of phishing sites. The machine learning model is trained on a dataset containing labeled examples of both phishing and legitimate websites, allowing the system to learn the patterns that distinguish the two. The resulting model can then be used to automatically classify new URLs in real-time, providing an effective way to combat phishing and protect users from online threats.

The project leverages the following technologies:

**1.	Python Libraries:**
•	**Pandas**: For data manipulation and preprocessing.
•	**NumPy**: For numerical operations.
•	**Seaborn & Matplotlib**: For data visualization, including histograms, heatmaps, ROC curves, and precision-recall curves.
•	**Scikit-learn**: For machine learning models, model evaluation, and hyperparameter tuning. Specifically, we use: 
RandomForestClassifier, DecisionTreeClassifier, and XGBClassifier for classification tasks.
**GridSearchCV** for hyperparameter tuning.
**StandardScaler** for feature scaling.
**VotingClassifier** for combining multiple models into an ensemble.
roc_curve, auc, precision_recall_curve for evaluation.
•	**XGBoost**: For training an efficient gradient-boosted tree model, which is known for its high performance.
•	**URL Parsing**: For extracting features from URLs, including domain information, query components, and the presence of suspicious terms.

**2.	Machine Learning Algorithms:**
•	**Decision Trees**: A simple, interpretable classification model that splits the data based on features.
•	**Random Forests**: An ensemble of decision trees that improves classification accuracy by averaging predictions.
•	**XGBoost**: A highly efficient, scalable machine learning library for gradient boosting, known for winning many Kaggle competitions.
•	**Ensemble Voting Classifier**: A model that combines the predictions of Decision Tree, Random Forest, and XGBoost to improve overall prediction accuracy.

**3.	Model Evaluation:**
•	**Confusion Matrix**: To measure the performance of the classification models.
•	**Classification Report**: To evaluate precision, recall, and F1-score for each model.
•	**ROC Curve and AUC**: To assess the ability of the models to distinguish between phishing and legitimate websites.
•	**Precision-Recall Curve**: To evaluate the performance of models, especially for imbalanced datasets.
