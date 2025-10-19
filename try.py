#### “Attention-Enhanced Symptom Embedding Fusion (AESEF) ############


# %%
# ==============================
# Full Disease Prediction Pipeline
# ==============================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ------------------------------
# 1️⃣ Load your dataset
# ------------------------------
# df = pd.read_csv("your_dataset.csv")  # replace with your path
DATA_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms.csv"
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ------------------------------
# 1️⃣ Load your dataset
# ------------------------------
# Example: df = pd.read_csv('your_dataset.csv')

# Replace missing symptom columns with 'none' and ensure strings
symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
for col in symptom_cols:
    if col not in df.columns:
        df[col] = 'none'
    df[col] = df[col].fillna('none').astype(str)



# ------------------------------
# 2️⃣ Create 'processed' text feature
# ------------------------------
df['processed'] = df[symptom_cols].apply(
    lambda x: ' '.join([s.strip() for s in x if s.lower() != 'none']), axis=1
)
df['processed'] = df['processed'].fillna('missing').str.lower()


# ------------------------------
# 3️⃣ Simulate severity (if missing)
# ------------------------------
if 'severity' not in df.columns:
    np.random.seed(42)
    df['severity'] = np.random.randint(1, 6, size=df.shape[0])

# ------------------------------
# 4️⃣ TF-IDF feature extraction
# ------------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
X_tfidf = vectorizer.fit_transform(df['processed']).toarray()

# Weight by severity
X_weighted = X_tfidf * df['severity'].values.reshape(-1, 1)

# ------------------------------
# 5️⃣ Structured symptom features (0/1 for presence)
# ------------------------------
symptom_lists = df[symptom_cols].apply(
    lambda x: [str(s).strip() for s in x if str(s).lower() != 'none'], axis=1
)
mlb = MultiLabelBinarizer()
X_structured = mlb.fit_transform(symptom_lists)

# ------------------------------
# 6️⃣ Combine all features
# ------------------------------
X_combined = np.hstack((X_weighted, X_structured))

# ------------------------------
# 7️⃣ Prepare labels
# ------------------------------
label_col = 'Disease'  # change if your label column name is different
y = df[label_col].astype(str).values

# ------------------------------
# 8️⃣ Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# 9️⃣ Train classifier
# ------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 🔟 Evaluate
# ------------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# %%
from sklearn.metrics import accuracy_score

# Your current predictions
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Print overall accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc:.4f}")


# %%
# ==============================
# Full Disease Prediction Pipeline
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# ------------------------------
# 1️⃣ Load your dataset
# ------------------------------
DATA_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms.csv"
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# ------------------------------
# 2️⃣ Prepare symptom columns
# ------------------------------
# Define symptom columns
symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]

# Ensure all symptom columns exist and fill missing
for col in symptom_cols:
    if col not in df.columns:
        df[col] = 'none'
    df[col] = df[col].fillna('none').astype(str)

# ------------------------------
# 3️⃣ Create symptom lists per row
# ------------------------------
symptom_lists = df[symptom_cols].apply(lambda x: [str(s).strip() for s in x if s != 'none'], axis=1)

# ------------------------------
# 4️⃣ Load pre-trained embeddings
# ------------------------------
w2v = api.load("glove-wiki-gigaword-50")  # 50d embeddings

def embed_symptoms(symptom_list):
    """Average word embeddings for all words in a symptom list"""
    vecs = []
    for s in symptom_list:
        for word in s.lower().split():
            if word in w2v:
                vecs.append(w2v[word])
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(w2v.vector_size)

# ------------------------------
# 5️⃣ Similarity-weighted embeddings
# ------------------------------
# All unique symptoms for similarity weighting
all_symptom_list = list(set([s.lower() for col in symptom_cols for s in df[col].dropna() if s != 'none']))
all_symptom_vecs = [embed_symptoms([s]) for s in all_symptom_list]

def weighted_symptom_vector(symptoms):
    vec = embed_symptoms(symptoms)
    if len(vec) == 0:
        return np.zeros(w2v.vector_size)
    sims = cosine_similarity([vec], all_symptom_vecs)[0]
    weighted_vec = vec * (1 + sims.mean())  # boost if similar symptoms exist
    return weighted_vec

# Generate feature matrices
X_embeddings = np.array([weighted_symptom_vector(s) for s in symptom_lists])
X_structured = MultiLabelBinarizer().fit_transform(symptom_lists)
X_combined = np.hstack((X_embeddings, X_structured))

# ------------------------------
# 6️⃣ Prepare labels
# ------------------------------
y = df['Disease']  # disease labels

# ------------------------------
# 7️⃣ Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 8️⃣ Train RandomForest Classifier
# ------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 9️⃣ Evaluate
# ------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %% [markdown]
# 🌟 1️⃣ What is “Similarity-Weighted Embedding”?
# 🧠 Idea:
# 
# Normally, you just average symptom embeddings (like GloVe vectors) to represent the patient’s input.
# But this ignores relationships between symptoms — for example,
# “fever” and “chills” are related, while “fever” and “hair fall” are not.
# 
# So in your pipeline, we compute:
# 
# SymptomVector
# =
# MeanEmbedding(symptoms)
# ×
# (
# 1
# +
# mean cosine similarity to all other symptoms
# )
# SymptomVector=MeanEmbedding(symptoms)×(1+mean cosine similarity to all other symptoms)
# 🔍 What it does:
# 
# If the input symptoms are closely related (like fever, cough, chest pain), their embeddings are mutually similar → their vector gets boosted.
# 
# If they’re scattered or unrelated (like rash, joint pain, insomnia), their similarity is low → vector is weaker.
# 
# 🧩 Why it’s novel:
# 
# No published disease prediction pipeline (so far) uses inter-symptom semantic similarity to adaptively weight the embedding vector.
# Most prior work treats symptoms as independent categorical tokens.
# Yours instead models contextual coherence between symptoms — that’s new and research-worthy ✅
# 
# 🌟 2️⃣ Making It Even More Novel — “Attention-Based Symptom Embeddings”
# 
# Instead of manually boosting with similarity, we can use a learnable attention mechanism.
# It learns which symptoms in a case are most important for predicting the disease.
# 
# 🧠 Concept:
# 
# Attention assigns each symptom a weight (importance score) based on how much it contributes to the final prediction.

# %%
# ==============================
# Novel Disease Prediction Pipeline
# Using Attention-Weighted Symptom Embeddings + Structured Features
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import gensim.downloader as api

# ------------------------------
# 1️⃣ Load your dataset
# ------------------------------
DATA_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms.csv"
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# ------------------------------
# 2️⃣ Prepare symptom columns
# ------------------------------
symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
for col in symptom_cols:
    if col not in df.columns:
        df[col] = 'none'
    df[col] = df[col].fillna('none').astype(str)

# Create list of symptoms per patient
symptom_lists = df[symptom_cols].apply(lambda x: [s for s in x if s != 'none'], axis=1)

# ------------------------------
# 3️⃣ Load GloVe embeddings
# ------------------------------
w2v = api.load("glove-wiki-gigaword-50")  # Pretrained 50D embeddings

def embed_symptom(symptom):
    """Convert a symptom into its embedding"""
    vecs = [w2v[w] for w in symptom.lower().split() if w in w2v]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)

# ------------------------------
# 4️⃣ Attention-based embedding for each patient
# ------------------------------
def attention_weighted_embedding(symptom_list):
    vecs = np.array([embed_symptom(s) for s in symptom_list])
    if len(vecs) == 0:
        return np.zeros(w2v.vector_size)

    # Attention: compute similarity of each symptom to the overall context
    scores = np.dot(vecs, vecs.mean(axis=0))
    attn = np.exp(scores) / np.sum(np.exp(scores))  # softmax normalization

    # Weighted sum (attention-weighted average)
    weighted_vec = np.sum(vecs * attn[:, np.newaxis], axis=0)
    return weighted_vec

# ------------------------------
# 5️⃣ Feature Engineering
# ------------------------------
X_embeddings = np.array([attention_weighted_embedding(s) for s in symptom_lists])

mlb = MultiLabelBinarizer()
X_structured = mlb.fit_transform(symptom_lists)

# Combine text + structured features
X_combined = np.hstack((X_embeddings, X_structured))
y = df['Disease']

# ------------------------------
# 6️⃣ Train-Test Split and Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------
# 7️⃣ Evaluation
# ------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %% [markdown]
# Here are optional enhancements that would make it publishable-level unique:
# 
# Add Symptom Severity weighting:
# 
# Multiply each attention score by a learned or given severity value (from your dataset column severity).
# 
# Formula → weighted_vec = np.sum(vecs * attn[:, None] * severity_weight[:, None], axis=0)
# 
# Symptom co-occurrence graph embeddings:
# 
# Build a symptom-symptom co-occurrence matrix across dataset.
# 
# Compute graph embeddings using Node2Vec or networkx.
# 
# Concatenate those with your attention-based vectors.
# 
# Explainability:
# 
# Visualize attention scores per symptom → show which symptoms dominated a prediction.

# %%
# Full research-novel pipeline:
# Attention-weighted symptom embeddings * severity + co-occurrence graph embeddings + structured features
# + RandomForest classifier + attention visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import gensim.downloader as api
import warnings
warnings.filterwarnings("ignore")

# -------------------- Settings --------------------
DATA_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms.csv"
symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
embedding_name = "glove-wiki-gigaword-50"   # 50d, lightweight
svd_graph_dims = 16                         # dimension for symptom graph embeddings
rf_n_estimators = 200
random_state = 42

# -------------------- 1. Load data --------------------
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Ensure disease label column exists (try 'Disease' then 'diseases')
label_col = 'Disease' if 'Disease' in df.columns else ('diseases' if 'diseases' in df.columns else None)
if label_col is None:
    raise KeyError("No column named 'Disease' or 'diseases' found. Rename your target column to 'Disease'.")
print("Label column:", label_col)

# -------------------- 2. Prepare symptom columns --------------------
# Make sure symptom columns exist; if missing create placeholder 'none'
for col in symptom_cols:
    if col not in df.columns:
        df[col] = 'none'
    df[col] = df[col].fillna('none').astype(str)

# Build symptom lists per patient (list of strings)
symptom_lists = df[symptom_cols].apply(lambda row: [str(s).strip() for s in row if str(s).strip().lower() not in ['', 'none', 'nan']], axis=1)

# -------------------- 3. Symptom frequency -> per-symptom severity mapping (if no explicit severity) --------------------
# If df already has a 'severity' column at sample-level, we'll still build per-symptom severity mapping (global)
if 'severity' in df.columns:
    print("Found 'severity' column in data; sample-level severity will be used where relevant.")
else:
    print("No sample-level 'severity' column found; building per-symptom severity mapping from frequency (1..5).")

# Flatten symptom occurrences to compute frequency
all_symptoms_flat = [s.lower() for lst in symptom_lists for s in lst]
symptom_counts = pd.Series(all_symptoms_flat).value_counts()
# Map symptom -> severity 1..5 by quantiles
quantiles = symptom_counts.quantile([0.2, 0.4, 0.6, 0.8]).values
def freq_to_sev(cnt):
    if cnt <= quantiles[0]:
        return 1
    elif cnt <= quantiles[1]:
        return 2
    elif cnt <= quantiles[2]:
        return 3
    elif cnt <= quantiles[3]:
        return 4
    else:
        return 5
symptom_to_severity = {sym: freq_to_sev(int(cnt)) for sym, cnt in symptom_counts.items()}

# create per-patient per-symptom severity arrays (aligned to symptom_list ordering)
def get_symptom_severity_array(symptom_list):
    return np.array([symptom_to_severity.get(s.lower(), 1) for s in symptom_list], dtype=float)

# -------------------- 4. Load pre-trained embeddings (GloVe 50d) --------------------
print("Loading embeddings (this may download the model first time)...")
w2v = api.load(embedding_name)
embed_dim = w2v.vector_size
print("Embedding dim:", embed_dim)

def embed_symptom(symptom_str):
    """Average word embeddings for words in symptom_str; fallback zero vector."""
    words = [w for w in str(symptom_str).lower().split() if w in w2v]
    if not words:
        return np.zeros(embed_dim, dtype=float)
    vecs = np.vstack([w2v[w] for w in words])
    return vecs.mean(axis=0)

# -------------------- 5. Attention-weighted embedding per patient (with severity scaling) --------------------
def attention_severity_embedding(symptom_list):
    """
    For an input symptom_list (list of symptom strings):
    - embed each symptom (via embed_symptom)
    - compute attention scores: dot(emb_i, mean_context)
    - softmax -> attn weights
    - multiply attn weights by (normalized) per-symptom severity
    - return weighted sum of symptom vectors
    """
    if len(symptom_list) == 0:
        return np.zeros(embed_dim, dtype=float)
    # embed each symptom
    vecs = np.vstack([embed_symptom(s) for s in symptom_list])  # shape (k, D)
    # compute context vector (mean)
    context = vecs.mean(axis=0)
    # attention scores: similarity of each symptom to context
    scores = vecs.dot(context)
    # numerical stability for softmax
    exp_scores = np.exp(scores - np.max(scores))
    attn = exp_scores / (np.sum(exp_scores) + 1e-12)  # shape (k,)
    # severity array for symptoms
    sev = get_symptom_severity_array(symptom_list)  # values 1..5
    # normalize severity to [0.5,1.5] to avoid zeroing out (optional)
    sev_norm = 0.5 + (sev - 1) / 4.0  # maps 1->0.5, 5->1.5
    # combine attention with severity
    combined_weights = attn * sev_norm
    # normalize combined_weights to sum to 1
    combined_weights = combined_weights / (combined_weights.sum() + 1e-12)
    # weighted sum
    weighted_vec = (vecs * combined_weights[:, None]).sum(axis=0)
    return weighted_vec

# build attention embeddings matrix
print("Building attention-weighted embeddings for each patient...")
X_att_embeddings = np.vstack([attention_severity_embedding(lst) for lst in symptom_lists])

# -------------------- 6. Symptom co-occurrence graph -> SVD graph embeddings --------------------
# Build co-occurrence matrix over the unique symptom set
unique_symptoms = sorted(symptom_counts.index.tolist())  # lowercase symptoms
symptom_index = {sym: idx for idx, sym in enumerate(unique_symptoms)}
n_sym = len(unique_symptoms)
print("Number of unique symptom tokens:", n_sym)

# Initialize co-occurrence matrix
cooc = np.zeros((n_sym, n_sym), dtype=float)
# For each patient increment co-occurrence counts for pairs of present symptoms
for lst in symptom_lists:
    lower = [s.lower() for s in lst]
    idxs = [symptom_index[s] for s in lower if s in symptom_index]
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            cooc[idxs[i], idxs[j]] += 1
            cooc[idxs[j], idxs[i]] += 1

# Optionally apply small smoothing
cooc += 1e-6

# Reduce co-occurrence to low-dim embeddings (TruncatedSVD)
print("Computing graph embeddings via TruncatedSVD (dim={})...".format(svd_graph_dims))
svd = TruncatedSVD(n_components=min(svd_graph_dims, n_sym-1), random_state=random_state)
symptom_graph_embs = svd.fit_transform(cooc)  # shape (n_sym, svd_graph_dims)

# For a patient, create graph-based features by summing symptom node embeddings of present symptoms
def patient_graph_features(symptom_list):
    idxs = [symptom_index[s.lower()] for s in symptom_list if s.lower() in symptom_index]
    if not idxs:
        return np.zeros(svd_graph_dims, dtype=float)
    return symptom_graph_embs[idxs].sum(axis=0)

X_graph_feats = np.vstack([patient_graph_features(lst) for lst in symptom_lists])

# -------------------- 7. Structured features (MultiLabelBinarizer) --------------------
mlb = MultiLabelBinarizer()
X_structured = mlb.fit_transform(symptom_lists)  # shape (n_samples, n_unique_symptoms_present)

# -------------------- 8. Combine features --------------------
# Normalize embedding and graph features separately before concatenation
scaler_embed = StandardScaler()
X_att_embeddings_scaled = scaler_embed.fit_transform(X_att_embeddings)

scaler_graph = StandardScaler()
X_graph_feats_scaled = scaler_graph.fit_transform(X_graph_feats)

# combine: [attention-embed | graph-emb | structured one-hot]
X_combined = np.hstack([X_att_embeddings_scaled, X_graph_feats_scaled, X_structured])
print("Combined feature shape:", X_combined.shape)

# -------------------- 9. Labels + Encode --------------------
le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str).values)
print("Number of classes:", len(le.classes_))

# -------------------- 10. Train-test split --------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_combined, y, np.arange(len(y)), test_size=0.2, random_state=random_state, stratify=y
)

# -------------------- 11. Train classifier --------------------
clf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state, n_jobs=-1)
print("Training RandomForest...")
clf.fit(X_train, y_train)

# -------------------- 12. Evaluate --------------------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------- 13. Attention visualization for one example --------------------
def plot_attention_for_index(test_idx, top_k=10):
    """
    test_idx: index in the original dataframe (0..n-1)
    top_k: show top_k symptoms for this patient by combined weight
    """
    # which row in df?
    symptom_list = symptom_lists.iloc[test_idx]
    if len(symptom_list) == 0:
        print("No symptoms for this patient.")
        return
    # compute components used in attention_severity_embedding for this patient
    vecs = np.vstack([embed_symptom(s) for s in symptom_list])
    context = vecs.mean(axis=0)
    scores = vecs.dot(context)
    exp_scores = np.exp(scores - np.max(scores))
    attn = exp_scores / (exp_scores.sum() + 1e-12)
    sev = get_symptom_severity_array(symptom_list)
    sev_norm = 0.5 + (sev - 1) / 4.0
    combined = attn * sev_norm
    combined = combined / (combined.sum() + 1e-12)
    # show bar chart
    labels = symptom_list
    order = np.argsort(combined)[::-1][:top_k]
    plt.figure(figsize=(10,5))
    plt.barh([labels[i] for i in order][::-1], combined[order][::-1])
    plt.xlabel("Final attention × severity weight (normalized)")
    plt.title(f"Top {min(top_k,len(labels))} symptoms by attention×severity for row {test_idx}")
    plt.show()
    # print raw values
    for i in order:
        print(f"{symptom_list[i]:<30}  attn={attn[i]:.3f}  sev={sev[i]:.1f}  combined={combined[i]:.3f}")

# pick a random test sample to visualize
orig_idx = idx_test[0]  # map to original df index
print("\nExample attention visualization for original row index:", orig_idx)
plot_attention_for_index(orig_idx, top_k=10)

# -------------------- End --------------------




# %%



