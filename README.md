# Symptom2Disease: An Explainable AI Framework for Disease Prediction

Overview

Symptom2Disease is an AI-driven framework designed to predict diseases from symptom descriptions using Natural Language Processing (NLP) and Deep Learning.
It integrates traditional vectorization techniques like TF-IDF, advanced Word2Vec/SpaCy embeddings, and Explainable AI (XAI) methods such as LIME and SHAP to make the predictions both accurate and interpretable.

Features

TF-IDF + Neural Network for baseline classification

Word2Vec/SpaCy Embeddings + Deep Neural Network for semantic learning

K-Means Clustering for grouping similar symptom patterns

Explainability using LIME & SHAP to visualize feature (symptom) importance

Confusion Matrix & Accuracy Reports for performance evaluation

Interactive Prediction Demo for testing new symptom inputs

Project Architecture
Data Preprocessing → Feature Extraction (TF-IDF / Embeddings)
                   → Model Training (Neural Network)
                   → Clustering Analysis (K-Means)
                   → Explainability (LIME, SHAP)
                   → Evaluation (Confusion Matrix, Accuracy)

Dataset

Source: Kaggle “Disease Symptoms Dataset” (41 diseases × 132 symptoms)

Columns:

Disease – target label

all_symptoms – preprocessed symptom text per record

Installation
# Clone the repository
git clone https://github.com/<your-username>/Symptom2Disease.git
cd Symptom2Disease

# Install dependencies
pip install -r requirements.txt

requirements.txt
pandas
numpy
scikit-learn
tensorflow
spacy
gensim
lime
shap
matplotlib
seaborn

Implementation Steps
# 1. TF-IDF + K Means + Neural Network
# Convert symptoms to TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(final2_df['all_symptoms']).toarray()

# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_tfidf)

Output:

<img width="775" height="547" alt="image" src="https://github.com/user-attachments/assets/b44df7fe-a052-4f03-8dcf-3c3642e9884d" />

5 distinct clusters visualized using t-SNE/UMAP

Similar symptom sets grouped together


# Train neural network model
model_tfidf.fit(X_train, y_train, epochs=20, batch_size=32)


Output:

<img width="691" height="539" alt="image" src="https://github.com/user-attachments/assets/f56486d1-bbce-47f7-aed3-ed8c9a13359a" />


Validation Accuracy: ~90–93%
Test Accuracy: ~0.92

Visual Outputs

Accuracy Curve: Shows steady improvement, no overfitting.

<img width="725" height="571" alt="image" src="https://github.com/user-attachments/assets/25f751e2-f00b-4c9a-9d19-05de75b213ba" />

Loss Curve: Converges smoothly after ~10–15 epochs.

<img width="716" height="581" alt="image" src="https://github.com/user-attachments/assets/6c35f8cb-d0a0-4567-987d-0c321709d214" />

Classification Metrics

Metric	Value

Precision	0.91

Recall	0.90

F1-Score	0.91

Confusion Metrics

<img width="767" height="541" alt="image" src="https://github.com/user-attachments/assets/52a65779-afb6-4245-88c5-642fce814436" />





# 2. Word2Vec / SpaCy Embeddings + Deep Neural Network
# Convert each symptom set into dense vectors
X_spacy = np.array([nlp(text).vector for text in final2_df['all_symptoms']])

# Train model
model_embed.fit(X_train_embed, y_train_embed, epochs=20)


Output:

<img width="588" height="107" alt="image" src="https://github.com/user-attachments/assets/f101ee8b-0e30-44c1-a666-57594204d7f8" />
<img width="606" height="86" alt="image" src="https://github.com/user-attachments/assets/44b75c24-b0cb-493b-96a2-9c91a6673ad6" />


<img width="522" height="109" alt="image" src="https://github.com/user-attachments/assets/eea9a5a6-656f-4367-9c77-cd7b9bab225c" />

Validation Accuracy: ~94–96%
Predicted Disease: 'Pneumonia' and 'Brain Hemorrhage'

<img width="731" height="576" alt="image" src="https://github.com/user-attachments/assets/2d09a59e-eae5-40df-bca1-520d83b162c2" />
<img width="726" height="574" alt="image" src="https://github.com/user-attachments/assets/66c6e187-ce57-4ff8-a826-68bafa4a59bf" />

Confusion Matrix 

<img width="758" height="532" alt="image" src="https://github.com/user-attachments/assets/f678daea-ee80-4eca-bbe5-c55f499e07a9" />
<img width="640" height="85" alt="image" src="https://github.com/user-attachments/assets/62b99ee2-07a5-400d-99f3-30a0a200265e" />


# 3. Compare Accuracy of TF-IDF Vs Word2vec Emebedding
   
   <img width="381" height="75" alt="image" src="https://github.com/user-attachments/assets/04425e2d-1f00-4025-a835-9a3d1dea1aa2" />

from sklearn.metrics import confusion_matrix, classification_report

<img width="1012" height="839" alt="image" src="https://github.com/user-attachments/assets/0a03953b-ec78-438f-95cf-61f1d075ae44" />



y_pred = model.predict(X_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))


Output:

<img width="1302" height="1106" alt="image" src="https://github.com/user-attachments/assets/a71334e7-37e3-4cfa-af6a-d3e23592ccad" />


Accuracy: 0.94

Precision: 0.93

Recall: 0.95

F1-score: 0.94

Confusion matrix visualized for overall model performance.
 

# 4. Explainability (LIME + SHAP)
# LIME Explanation
exp = explainer.explain_instance(text_instance, model.predict_proba)
exp.show_in_notebook()

<img width="920" height="77" alt="image" src="https://github.com/user-attachments/assets/fde8eba4-dbf0-4cbe-af11-6f95aa1a457f" />

LIME chart showing which symptoms most influenced prediction

<img width="1779" height="119" alt="image" src="https://github.com/user-attachments/assets/9a2a44f9-90be-4a5d-adb2-5fbbefa5a961" />

Prediction Confidence 

<img width="460" height="103" alt="image" src="https://github.com/user-attachments/assets/8c9528a7-67b0-4a28-84c9-071ed59d89f0" />


# SHAP Explanation
shap_values = explainer(sample_data)
shap.plots.bar(shap_values[0])


Output:

<img width="789" height="940" alt="image" src="https://github.com/user-attachments/assets/65df489d-7687-4530-a22b-b4687d8dfab3" />

SHAP bar plot ranking features by contribution

<img width="790" height="1113" alt="image" src="https://github.com/user-attachments/assets/41f5d415-e959-4d49-93d2-fc2688b1bda2" />


# 6. Model Evaluation


	
Results Summary
Model Type	Feature Representation	Accuracy	

Neural Network + TF-IDF	Sparse BoW features	~92%	Fast baseline

Neural Network + Word2Vec/SpaCy	Dense semantic embeddings	~95%	Captures context

K-Means (TF-IDF)	Clustering of symptoms	—	For unsupervised grouping

LIME + SHAP	—	—	Explainability layer

Key Insights

SpaCy embeddings outperform TF-IDF for semantic understanding.

Explainability (LIME/SHAP) builds trust and interpretability for medical users.

The framework can be extended to real-world datasets for AI-assisted diagnosis.

🧑‍💻 Author

Chaitali Jain
B.Tech (Engineering Student)
AI & Data Science Enthusiast

📜 License

MIT License © 2025 Chaitali Jain
