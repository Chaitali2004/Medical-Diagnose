# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

final2_df = pd.read_csv("C:\\Users\\CHAITALI JAIN\\Desktop\\database for eds\\DiseaseAndSymptoms.csv")

# %%
final2_df.replace('0', pd.NA, inplace=True)
final2_df.describe()

# %%
all_unique_symptoms = set()
for i in range(1, 18):
    column_name = f'Symptom_{i}'
    unique_symptoms = final2_df[column_name].unique()
    all_unique_symptoms.update(unique_symptoms)

print(f"Total number of unique symptoms across all columns: {len(all_unique_symptoms)}")


print("Number of diseases that can be identified ",len(final2_df['Disease'].unique()))

# %%
# List of all symptom columns
symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]

# Join symptoms into a single text string for each disease
final2_df['all_symptoms'] = final2_df[symptom_cols].fillna('').agg(' '.join, axis=1)

# Check result
print(final2_df[['Disease', 'all_symptoms']].head())


# %% [markdown]
# # The next step is TF-IDF vectorization then K means Clustering 
# Converts symptoms → TF-IDF features.
# 
# Groups them into k=41 clusters (same as number of diseases).
# 
# Adds a cluster column to your dataframe.
# 
# Plots the clusters in 2D (using PCA).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(final2_df['all_symptoms'])

print("TF-IDF matrix shape:", X.shape)

# === Step 2: KMeans Clustering ===
k = 41  # number of diseases (clusters)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
final2_df['cluster'] = kmeans.fit_predict(X)

# === Step 3: Compare Clusters with Actual Diseases ===
print(final2_df[['Disease', 'cluster']].head(20))


# %% [markdown]
#  === Visualization (2D with PCA) ===
# 

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(14,10))
sns.scatterplot(x=reduced[:,0], y=reduced[:,1],
                hue=final2_df['cluster'], 
                palette='tab20', s=80, alpha=0.8, edgecolor='k')

plt.title("Disease Symptom Clusters (PCA Reduced to 2D)", fontsize=16, weight='bold')
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)

# Legend outside the plot
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=9)

plt.tight_layout()
plt.show()


# %% [markdown]
# # Training thorugh deep neural network 
# What this does
# 
# Learns a disease prediction model using symptom text
# 
# Outputs test accuracy (you can also plot training vs validation loss/accuracy)
# 
# Allows you to input custom symptoms and predict a disease

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# === Step 1: Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(final2_df['Disease'])
y_categorical = to_categorical(y, num_classes=len(label_encoder.classes_))

# === Step 2: TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=2000)  # keep all symptoms
X_tfidf = vectorizer.fit_transform(final2_df['all_symptoms']).toarray()

# === Step 3: Train-test split ===
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y_categorical, test_size=0.2, random_state=42, stratify=y
)

# === Step 4: Neural Network ===
model_tfidf = Sequential([
    Dense(256, input_shape=(X_tfidf.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model_tfidf.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Step 5: Train Model ===
history = model_tfidf.fit(
    X_train_tfidf, y_train_tfidf, 
    validation_data=(X_test_tfidf, y_test_tfidf),
    epochs=15, 
    batch_size=32, 
    verbose=1
)

# === Step 6: Evaluate ===
loss_tfidf, acc_tfidf = model_tfidf.evaluate(X_test_tfidf, y_test_tfidf, verbose=0)
print(f"TF-IDF Test Accuracy: {acc_tfidf:.2f}")

# === Step 7: Detailed Report ===
y_pred_probs = model_tfidf.predict(X_test_tfidf)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_tfidf, axis=1)

print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


# === Step 8: Predict Example ===
example_symptoms = ["fever cough headache fatigue"]
example_features = vectorizer.transform(example_symptoms).toarray()
prediction = model_tfidf.predict(example_features)
predicted_disease = label_encoder.inverse_transform([prediction.argmax()])
print("Predicted Disease:", predicted_disease[0])


# %%
# === Try new symptom examples ===
test_cases = [
    "itching skin_rash nodal_skin_eruptions",
    "abdominal_pain nausea vomiting yellowing_of_eyes",
    "chest_pain breathlessness sweating fatigue",
    "joint_pain back_pain stiffness",
    "high_fever headache chills muscle_pain"
]

for case in test_cases:
    features = vectorizer.transform([case]).toarray()
    prediction = model_tfidf.predict(features)
    predicted_disease = label_encoder.inverse_transform([prediction.argmax()])
    print(f"Symptoms: {case}")
    print(f"Predicted Disease: {predicted_disease[0]}\n")


# %% [markdown]
# Plot Accuracy and Loss Curves

# %%
import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% [markdown]
# Classification Report
#  
# It gives us Precision, Recall, F-1-Score, Support

# %%
from sklearn.metrics import classification_report
import numpy as np

# Predictions on test set
y_pred = model_tfidf.predict(X_test_tfidf)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_tfidf, axis=1)

# Report
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))


# %% [markdown]
# # Spacy (pre trained emedding model) + Neural Network 
# SEMNATIC MODEL
# 
# We use Word2Vec to convert symptoms (text) into dense numerical vectors that capture semantic meaning, so the model can understand relationships between symptoms beyond simple word counts (like in TF-IDF).

# %%
import spacy
nlp = spacy.load("en_core_web_md")

def get_spacy_vector(sentence):
    doc = nlp(sentence)
    return doc.vector

X_spacy = np.array([get_spacy_vector(s) for s in final2_df['all_symptoms']])
print("Shape:", X_spacy.shape)


# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Features (symptom embeddings)
X = X_spacy   # or X_word2vec if using gensim

# Labels (diseases)
le = LabelEncoder()
y = le.fit_transform(final2_df['Disease'])
y = to_categorical(y)

# Split into train & test
X_train, X_test_embed, y_train, Y_test_embed = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network
model_embed = Sequential()
model_embed.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model_embed.add(Dropout(0.3))
model_embed.add(Dense(128, activation='relu'))
model_embed.add(Dropout(0.3))
model_embed.add(Dense(y.shape[1], activation='softmax'))

model_embed.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model_embed.fit(X_test_embed, Y_test_embed, validation_data=(X_test_embed, Y_test_embed), epochs=15, batch_size=32, verbose=1)

# Evaluate
loss, acc = model_embed.evaluate(X_test_embed, Y_test_embed, verbose=0)
print(f"Test Accuracy: {acc:.2f}")


# %%
# Example: new symptoms
new_symptom_text = "fever cough headache fatigue"

vec = get_spacy_vector(new_symptom_text).reshape(1, -1)
pred = model_embed.predict(vec)
pred_disease = le.inverse_transform([np.argmax(pred)])
print("Predicted Disease:", pred_disease[0])


# %%
# Another test case
new_symptom_text = "chest_pain shortness_of_breath dizziness"

vec = get_spacy_vector(new_symptom_text).reshape(1, -1)
pred = model_embed.predict(vec)
pred_disease = le.inverse_transform([np.argmax(pred)])
print("Predicted Disease:", pred_disease[0])


# %% [markdown]
# Vizualize Training Performance

# %%
import matplotlib.pyplot as plt

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training vs validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% [markdown]
# Comapre Accuracy of TF-IDF Vs Word2vec Emebedding 

# %%
# TF-IDF + NN result
loss_tfidf, acc_tfidf = model_tfidf.evaluate(X_test_tfidf, y_test_tfidf, verbose=0)

# Word2Vec/SpaCy + NN result
loss_embed, acc_embed = model_embed.evaluate(X_test_embed, Y_test_embed, verbose=0)

print(f"TF-IDF Model Accuracy: {acc_tfidf:.2f}")
print(f"Embedding Model Accuracy: {acc_embed:.2f}")


# %%
# Confusion Matrices for both
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict TF-IDF model
y_pred_tfidf = np.argmax(model_tfidf.predict(X_test_tfidf), axis=1)
y_true_tfidf = np.argmax(y_test_tfidf, axis=1)

# Predict Embedding model
y_pred_embed = np.argmax(model_embed.predict(X_test_embed), axis=1)
y_true_embed = np.argmax(Y_test_embed, axis=1)

# Confusion matrix - TFIDF
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_true_tfidf, y_pred_tfidf), annot=False, cmap="Blues")
plt.title("TF-IDF Model Confusion Matrix")
plt.show()

# Confusion matrix - Embeddings
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_true_embed, y_pred_embed), annot=False, cmap="Greens")
plt.title("Embedding Model Confusion Matrix")
plt.show()

# Classification reports
print("TF-IDF Model Report:\n", classification_report(y_true_tfidf, y_pred_tfidf))
print("Embedding Model Report:\n", classification_report(y_true_embed, y_pred_embed))


# %% [markdown]
# # Add Explainable AI with LIME 
# 
# A Neural Network is like a black box: it gives predictions but doesn’t say why.
# 
# In healthcare AI, it’s not enough to predict the disease — doctors need to know which symptoms influenced the prediction.
# A LIME visualization showing which symptoms influenced a prediction.
# LIME and SHAP give this transparency:

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(final2_df['all_symptoms'])

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(final2_df['Disease'])

# Train/test split
from sklearn.model_selection import train_test_split
X_train_vec, X_test_vec, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# One-hot encode target
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Define NN
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train_vec.toarray(), y_train_cat, 
                    epochs=10, batch_size=32,
                    validation_data=(X_test_vec.toarray(), y_test_cat))


# %% [markdown]
#  Step 2: Evaluate Model

# %%
loss, acc = model.evaluate(X_test_vec.toarray(), y_test_cat)
print(f"Test Accuracy: {acc:.2f}")


# %% [markdown]
#  Step 3: Explain Predictions with LIME

# %%
from lime.lime_text import LimeTextExplainer
import numpy as np

# Class names are the diseases
class_names = label_encoder.classes_

explainer = LimeTextExplainer(class_names=class_names)

# Example: pick one test sample
idx = 5
sample_text = final2_df['all_symptoms'].iloc[X_test_vec.indices[idx]]
print("Sample symptoms:", sample_text)
print("True disease:", class_names[y_test[idx]])

# Define a prediction function for LIME
predict_fn = lambda texts: model.predict(vectorizer.transform(texts).toarray())

# Explain the instance
exp = explainer.explain_instance(sample_text, predict_fn, num_features=6)
print(exp.as_list())



# %% [markdown]
# The weights are very small and negative. That usually means:
# 
# The model is not strongly confident about this prediction (possibly misclassified).
# Or the NN still needs more epochs / better preprocessing to learn stronger associations.
# 
# Next Steps -
# Check prediction confidence

# %%
probs = predict_fn([sample_text])[0]
predicted_class = class_names[np.argmax(probs)]
print("Predicted disease:", predicted_class)
print("Prediction confidence:", np.max(probs))


# %% [markdown]
# # Implementing the SHAP 
# This will give you both local explanations (per prediction) and global explanations (which symptoms matter overall for the model).

# %% [markdown]
# why use SHAP - Global + Local Explanations
# 
# Local: Like LIME, SHAP can explain a single prediction.
# 
# Global: It can also aggregate across all predictions to show which symptoms matter most overall.
# 
# Example: it might show that “fever” and “fatigue” are the top-2 symptoms driving predictions across all diseases.
# 
# Why run SHAP after LIME?
# 
# LIME helped you debug a few predictions → “is the model learning anything?”
# 
# SHAP will help you explain your model in the research paper → “here’s how symptoms influence diseases overall.”
# 
# In short:
# 
# LIME = local microscope 
# 
# SHAP = local + global + theory-backed telescope 

# %%
import pandas as pd
import shap

# Get feature names from TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()
# Convert text → vectors before SHAP
X_test_array = X_test_vec.toarray()
# Wrap train & test arrays into DataFrames
X_train_df = pd.DataFrame(X_train_vec.toarray(), columns=feature_names)
X_test_df = pd.DataFrame(X_test_array, columns=feature_names)

# Recreate SHAP explainer with DataFrame (keeps feature names)
explainer = shap.Explainer(model, X_train_df)

# Pick a few test samples
sample_data = X_test_df.sample(5, random_state=42)

# Compute SHAP values
shap_values = explainer(sample_data)

# Plot for first sample
shap.plots.bar(shap_values[0])


# %% [markdown]
# This will create a beeswarm plot (or a bar version) that tells you:
# 
# Which symptoms are most predictive across the whole dataset.
# How their presence/absence shifts disease predictions.

# %% [markdown]
# Next Step (Global Feature Importance with SHAP)
# 
# So far, we explained single samples. Next, let’s check global importance across all test data to see which symptoms generally matter the most in predicting diseases.

# %%
# Global explanation across the test set
shap_values = explainer(X_test_df)

# Summary plot - shows overall symptom importance
# Example: pick class index 0 (first disease in label encoder)
shap.summary_plot(shap_values[:, :, 0], X_test_df, feature_names=feature_names, plot_type="bar")



# %%
# Global explanation across the test set
shap_values = explainer(X_test_df)
# Bar plot version for better readability
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, plot_type="bar")


# %% [markdown]
# # Step 6: Evaluate Model Performance
# 
# Generate metrics:
# Accuracy, Precision, Recall, F1-score
# Confusion matrix (to see which diseases get mixed up)
# Plot these results for clarity.

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Predict on test data
y_pred_probs = model_tfidf.predict(X_test_vec)   # probabilities
y_pred_classes = y_pred_probs.argmax(axis=1)  # predicted class indices

# 2. Decode back to disease names
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# 3. Classification report
print("Classification Report:\n")
print(classification_report(y_test_labels, y_pred_labels))

# 4. Accuracy
print("Test Accuracy:", accuracy_score(y_test_labels, y_pred_labels))

# 5. Confusion Matrix
plt.figure(figsize=(14,10))
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

sns.heatmap(cm, 
            annot=False, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# %% [markdown]
# The above graph give us Precision, Recall, F1 per disease
# 
# Overall accuracy
# A heatmap confusion matrix (dark diagonal = good, off-diagonal = misclassifications)

# %% [markdown]
# # save model + vectorize 

# %%
import joblib

# Save model
model.save("disease_prediction_model.h5")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model, vectorizer, and encoder saved successfully!")


# %%
from tensorflow.keras.models import load_model

# Load back
loaded_model = load_model("disease_prediction_model.h5")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
loaded_encoder = joblib.load("label_encoder.pkl")

# Test prediction
sample = "itching skin_rash nodal_skin_eruptions"
X_sample = loaded_vectorizer.transform([sample])
y_pred = loaded_model.predict(X_sample).argmax(axis=1)
print("Predicted disease:", loaded_encoder.inverse_transform(y_pred)[0])



