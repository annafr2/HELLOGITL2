"""
Sentence Clustering Analysis using K-Means and KNN
This script demonstrates clustering sentences by semantic meaning
and testing classification with new sentences.

NO SKLEARN REQUIRED - All algorithms implemented from scratch!
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import seaborn as sns
from collections import Counter

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CUSTOM IMPLEMENTATIONS (No sklearn required!)
# ============================================================================

def normalize(vectors):
    """L2 normalization: scale each vector to unit length"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms

class KMeans:
    """K-Means clustering implementation from scratch"""
    def __init__(self, n_clusters=3, random_state=42, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.cluster_centers_ = None

    def _init_centroids(self, X, rng):
        """Initialize centroids randomly"""
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.cluster_centers_[i], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.cluster_centers_[i]
        return new_centroids

    def fit_predict(self, X):
        """Fit K-Means and return cluster labels"""
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        best_labels = None
        best_centers = None

        # Try multiple initializations
        for _ in range(self.n_init):
            centroids = self._init_centroids(X, rng)

            # Run K-Means iterations
            for _ in range(self.max_iter):
                self.cluster_centers_ = centroids
                labels = self._assign_clusters(X)
                new_centroids = self._update_centroids(X, labels)

                # Check convergence
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            # Calculate inertia (sum of squared distances)
            inertia = 0
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    inertia += np.sum((cluster_points - centroids[i])**2)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centroids

        self.cluster_centers_ = best_centers
        return best_labels

class KNeighborsClassifier:
    """KNN classifier implementation from scratch"""
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        """Predict labels for test data"""
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            # Get labels of k nearest neighbors
            k_labels = self.y_train[k_indices]
            # Return most common label
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)

def accuracy_score(y_true, y_pred):
    """Calculate accuracy"""
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix(y_true, y_pred):
    """Create confusion matrix"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix

def linear_sum_assignment(cost_matrix):
    """
    Simple greedy assignment for matching clusters to labels
    (Simplified version of Hungarian algorithm)
    """
    cost = -cost_matrix  # We want to maximize, so negate
    n = cost.shape[0]
    row_ind = []
    col_ind = []
    available_cols = list(range(cost.shape[1]))

    for i in range(n):
        best_col = None
        best_value = -np.inf
        for col in available_cols:
            if cost[i, col] > best_value:
                best_value = cost[i, col]
                best_col = col
        row_ind.append(i)
        col_ind.append(best_col)
        available_cols.remove(best_col)

    return np.array(row_ind), np.array(col_ind)

# ============================================================================
# PART 1: TRAINING DATA - 9 sentences in 3 categories
# ============================================================================

sentences = [
    # Animals (3 sentences)
    "The dog runs in the park",
    "Cats love to sleep all day",
    "Birds fly high in the sky",
    
    # Airplanes (3 sentences)
    "The airplane flies above clouds",
    "Jets travel at high speed",
    "Pilots control the aircraft carefully",
    
    # Cars (3 sentences)
    "The car drives on the highway",
    "Vehicles need regular maintenance",
    "Drivers must follow traffic rules"
]

# True labels (our original categorization)
true_labels = [
    "Animals", "Animals", "Animals",
    "Airplanes", "Airplanes", "Airplanes",
    "Cars", "Cars", "Cars"
]

print("="*70)
print("SENTENCE CLUSTERING ANALYSIS")
print("="*70)
print("\n📝 Training Sentences:")
for i, (sent, label) in enumerate(zip(sentences, true_labels)):
    print(f"{i+1}. [{label:>10}] {sent}")

# ============================================================================
# STEP 1: Convert sentences to vectors using Word2Vec
# ============================================================================

print("\n" + "="*70)
print("STEP 1: CONVERTING SENTENCES TO VECTORS (Word2Vec)")
print("="*70)

# Tokenize sentences
tokenized_sentences = [sent.lower().split() for sent in sentences]

# Train Word2Vec model (free, open-source)
# Parameters: vector_size=50, window=5, min_count=1
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4,
    epochs=100
)

print(f"✅ Word2Vec model trained successfully!")
print(f"   - Vocabulary size: {len(word2vec_model.wv)} words")
print(f"   - Vector dimensions: {word2vec_model.wv.vector_size}")

# Convert each sentence to a vector (average of word vectors)
def sentence_to_vector(sentence, model):
    """Convert a sentence to a vector by averaging word vectors"""
    words = sentence.lower().split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.wv.vector_size)
    return np.mean(word_vectors, axis=0)

sentence_vectors = np.array([
    sentence_to_vector(sent, word2vec_model) 
    for sent in sentences
])

print(f"\n📊 Sentence vectors shape: {sentence_vectors.shape}")
print(f"   (9 sentences × 50 dimensions)")

# ============================================================================
# STEP 2: Normalize vectors
# ============================================================================

print("\n" + "="*70)
print("STEP 2: NORMALIZING VECTORS")
print("="*70)

normalized_vectors = normalize(sentence_vectors)
print("✅ Vectors normalized (L2 norm)")
print(f"   Each vector now has unit length (norm = 1.0)")

# ============================================================================
# STEP 3: Apply K-Means clustering (K=3)
# ============================================================================

print("\n" + "="*70)
print("STEP 3: APPLYING K-MEANS CLUSTERING (K=3)")
print("="*70)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(normalized_vectors)

print("✅ K-Means clustering completed!")
print(f"\n🔍 K-Means cluster assignments:")
for i, (sent, true_label, km_label) in enumerate(zip(sentences, true_labels, kmeans_labels)):
    print(f"{i+1}. Cluster {km_label} | True: {true_label:>10} | {sent}")

# ============================================================================
# STEP 4: Evaluate K-Means clustering quality
# ============================================================================

print("\n" + "="*70)
print("STEP 4: EVALUATING K-MEANS CLUSTERING")
print("="*70)

# Map true labels to numeric values
label_mapping = {"Animals": 0, "Airplanes": 1, "Cars": 2}
true_numeric_labels = [label_mapping[label] for label in true_labels]

# Find the best mapping between K-Means clusters and true labels
# (using our custom implementations above)

# Create confusion matrix
conf_matrix = confusion_matrix(true_numeric_labels, kmeans_labels)

# Find optimal cluster-to-label mapping
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
cluster_to_label_map = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

# Remap K-Means labels to match true labels
remapped_kmeans_labels = [cluster_to_label_map[label] for label in kmeans_labels]

# Calculate accuracy
accuracy = accuracy_score(true_numeric_labels, remapped_kmeans_labels)
mistakes = 9 - int(accuracy * 9)

print(f"📊 Clustering Accuracy: {accuracy*100:.1f}%")
print(f"❌ Number of mistakes: {mistakes}/9")
print(f"✅ Correct classifications: {9-mistakes}/9")

# Analyze cluster composition
print(f"\n🔬 Cluster Analysis:")
label_names = ["Animals", "Airplanes", "Cars"]
for cluster_id in range(3):
    cluster_sentences = [sentences[i] for i in range(9) if kmeans_labels[i] == cluster_id]
    cluster_true_labels = [true_labels[i] for i in range(9) if kmeans_labels[i] == cluster_id]
    label_counts = Counter(cluster_true_labels)
    
    print(f"\n   Cluster {cluster_id}:")
    for sent in cluster_sentences:
        print(f"      - {sent}")
    print(f"   Composition: {dict(label_counts)}")

# ============================================================================
# VISUALIZATION 1: Cluster Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('K-Means Clustering Analysis', fontsize=18, fontweight='bold')

# Plot 1: Our Original Classification
ax1 = axes[0, 0]
colors_true = [plt.cm.Set2(label_mapping[label]) for label in true_labels]
for i, (label, color) in enumerate(zip(true_labels, colors_true)):
    ax1.scatter(i, 0, s=500, c=[color], alpha=0.7, edgecolors='black', linewidth=2)
    ax1.text(i, -0.15, f"S{i+1}", ha='center', fontsize=10, fontweight='bold')
    ax1.text(i, 0.15, label, ha='center', fontsize=8)

ax1.set_xlim(-0.5, 8.5)
ax1.set_ylim(-0.5, 0.5)
ax1.set_title('Our Original Classification', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sentence Number', fontsize=12)
ax1.set_yticks([])
ax1.grid(True, alpha=0.3)

# Plot 2: K-Means Classification
ax2 = axes[0, 1]
colors_kmeans = [plt.cm.Set2(label) for label in kmeans_labels]
for i, (km_label, color) in enumerate(zip(kmeans_labels, colors_kmeans)):
    ax2.scatter(i, 0, s=500, c=[color], alpha=0.7, edgecolors='black', linewidth=2)
    ax2.text(i, -0.15, f"S{i+1}", ha='center', fontsize=10, fontweight='bold')
    ax2.text(i, 0.15, f"Cluster {km_label}", ha='center', fontsize=8)

ax2.set_xlim(-0.5, 8.5)
ax2.set_ylim(-0.5, 0.5)
ax2.set_title('K-Means Classification', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sentence Number', fontsize=12)
ax2.set_yticks([])
ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy Bar Chart
ax3 = axes[1, 0]
categories = ['Correct', 'Incorrect']
values = [9-mistakes, mistakes]
colors_bar = ['#2ecc71', '#e74c3c']
bars = ax3.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Number of Sentences', fontsize=12)
ax3.set_title(f'Classification Accuracy: {accuracy*100:.1f}%', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 10)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}/9',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Plot 4: Confusion Matrix
ax4 = axes[1, 1]
conf_matrix_display = confusion_matrix(true_numeric_labels, remapped_kmeans_labels)
sns.heatmap(conf_matrix_display, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=label_names, yticklabels=label_names,
            cbar_kws={'label': 'Count'}, ax=ax4, linewidths=2, linecolor='black')
ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('K-Means Predicted Label', fontsize=12)
ax4.set_ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.savefig('kmeans_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: kmeans_analysis.png")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "="*70)
print("💡 INTERPRETATION")
print("="*70)

if accuracy >= 0.8:
    print("✅ K-Means did a GOOD job clustering!")
    print("   The algorithm found patterns similar to our categorization.")
    print("   Sentences with similar semantic meaning were grouped together.")
elif accuracy >= 0.5:
    print("⚠️  K-Means found a DIFFERENT pattern!")
    print("   The algorithm may have discovered alternative semantic similarities.")
    print("   This could mean it focused on different features than we did.")
else:
    print("❌ K-Means clustering was POOR.")
    print("   The algorithm struggled to find the patterns we expected.")

print("\n🔍 Possible reasons for K-Means decisions:")
print("   1. Word similarity: 'fly', 'high', 'sky' might connect airplanes and birds")
print("   2. Action patterns: 'runs', 'drives', 'flies' indicate movement")
print("   3. Context words: 'park', 'highway', 'clouds' create different contexts")

# ============================================================================
# PART 2: TESTING WITH NEW SENTENCES USING KNN
# ============================================================================

print("\n" + "="*70)
print("PART 2: TESTING WITH NEW SENTENCES (KNN)")
print("="*70)

# New test sentences
test_sentences = [
    "The elephant walks slowly",        # Should be Animals
    "Boeing planes are very large",     # Should be Airplanes
    "Trucks carry heavy loads"          # Should be Cars
]

test_true_labels = ["Animals", "Airplanes", "Cars"]

print("\n📝 Test Sentences:")
for i, sent in enumerate(test_sentences):
    print(f"{i+1}. {sent} (Expected: {test_true_labels[i]})")

# Convert test sentences to vectors
test_vectors = np.array([
    sentence_to_vector(sent, word2vec_model) 
    for sent in test_sentences
])
test_normalized = normalize(test_vectors)

print("\n✅ Test sentences converted to vectors and normalized")

# ============================================================================
# STEP 5: Train KNN on K-Means labels (not our original labels!)
# ============================================================================

print("\n" + "="*70)
print("STEP 5: TRAINING KNN CLASSIFIER")
print("="*70)

# Train KNN with K-Means labels (this is the key - we use what K-Means decided!)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(normalized_vectors, kmeans_labels)

print("✅ KNN trained on K-Means cluster assignments")
print("   Note: KNN learns from K-Means clusters, not our original labels!")

# Predict test sentences
knn_predictions = knn.predict(test_normalized)

print("\n🔍 KNN Predictions:")
for i, (sent, pred_cluster) in enumerate(zip(test_sentences, knn_predictions)):
    # Find which original sentences are in this cluster
    cluster_examples = [sentences[j] for j in range(9) if kmeans_labels[j] == pred_cluster]
    print(f"\n{i+1}. '{sent}'")
    print(f"   → Assigned to Cluster {pred_cluster}")
    print(f"   → This cluster contains:")
    for ex in cluster_examples:
        print(f"      • {ex}")

# ============================================================================
# STEP 6: Evaluate KNN predictions
# ============================================================================

print("\n" + "="*70)
print("STEP 6: EVALUATING KNN PREDICTIONS")
print("="*70)

# Map test true labels to numeric
test_numeric_true = [label_mapping[label] for label in test_true_labels]

# Remap KNN predictions to original labels using the same mapping
remapped_knn_predictions = [cluster_to_label_map[label] for label in knn_predictions]

# Calculate accuracy
knn_accuracy = accuracy_score(test_numeric_true, remapped_knn_predictions)
knn_mistakes = 3 - int(knn_accuracy * 3)

print(f"📊 KNN Accuracy: {knn_accuracy*100:.1f}%")
print(f"❌ Number of mistakes: {knn_mistakes}/3")
print(f"✅ Correct classifications: {3-knn_mistakes}/3")

print("\n🎯 Detailed Results:")
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
for i, (sent, true_label, pred_cluster) in enumerate(zip(test_sentences, test_true_labels, knn_predictions)):
    pred_label = reverse_label_mapping[cluster_to_label_map[pred_cluster]]
    status = "✅" if true_label == pred_label else "❌"
    print(f"{status} '{sent}'")
    print(f"   Expected: {true_label} | Predicted: {pred_label}")

# ============================================================================
# VISUALIZATION 2: KNN Results
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('KNN Classification Results', fontsize=18, fontweight='bold')

# Plot 1: Expected Classification
ax1 = axes[0, 0]
colors_expected = [plt.cm.Set2(label_mapping[label]) for label in test_true_labels]
for i, (label, color) in enumerate(zip(test_true_labels, colors_expected)):
    ax1.scatter(i, 0, s=600, c=[color], alpha=0.7, edgecolors='black', linewidth=2)
    ax1.text(i, -0.15, f"Test {i+1}", ha='center', fontsize=10, fontweight='bold')
    ax1.text(i, 0.15, label, ha='center', fontsize=9)

ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 0.5)
ax1.set_title('Expected Classification', fontsize=14, fontweight='bold')
ax1.set_xlabel('Test Sentence Number', fontsize=12)
ax1.set_yticks([])
ax1.grid(True, alpha=0.3)

# Plot 2: KNN Predictions
ax2 = axes[0, 1]
colors_knn = [plt.cm.Set2(cluster) for cluster in knn_predictions]
for i, (cluster, color) in enumerate(zip(knn_predictions, colors_knn)):
    ax2.scatter(i, 0, s=600, c=[color], alpha=0.7, edgecolors='black', linewidth=2)
    ax2.text(i, -0.15, f"Test {i+1}", ha='center', fontsize=10, fontweight='bold')
    ax2.text(i, 0.15, f"Cluster {cluster}", ha='center', fontsize=9)

ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(-0.5, 0.5)
ax2.set_title('KNN Predictions', fontsize=14, fontweight='bold')
ax2.set_xlabel('Test Sentence Number', fontsize=12)
ax2.set_yticks([])
ax2.grid(True, alpha=0.3)

# Plot 3: KNN Accuracy
ax3 = axes[1, 0]
categories = ['Correct', 'Incorrect']
values = [3-knn_mistakes, knn_mistakes]
colors_bar = ['#3498db', '#e67e22']
bars = ax3.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Number of Sentences', fontsize=12)
ax3.set_title(f'KNN Accuracy: {knn_accuracy*100:.1f}%', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 4)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}/3',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Plot 4: Comparison Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
KNN FOLLOWS K-MEANS CLUSTERING!

Key Insight:
• KNN was trained on K-Means labels
• KNN learns the pattern K-Means found
• NOT our original categorization

Results:
• K-Means Accuracy: {accuracy*100:.1f}%
• KNN Accuracy: {knn_accuracy*100:.1f}%

What does this mean?
{"✅ KNN successfully learned K-Means pattern" if knn_accuracy >= 0.66 else "⚠️ KNN struggled with K-Means pattern"}

The classification follows:
{"✅ Our original logic" if accuracy >= 0.8 else "❌ A different semantic pattern"}
"""

ax4.text(0.5, 0.5, summary_text, ha='center', va='center', 
         fontsize=12, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('knn_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: knn_results.png")

# ============================================================================
# FINAL CONCLUSIONS
# ============================================================================

print("\n" + "="*70)
print("🎓 FINAL CONCLUSIONS")
print("="*70)

print("\n1️⃣  K-MEANS CLUSTERING:")
if accuracy >= 0.8:
    print("   ✅ Found patterns matching our categorization")
    print("   ✅ Successfully separated animals, airplanes, and cars")
else:
    print("   ⚠️  Found different patterns than expected")
    print("   ⚠️  May have focused on linguistic features like:")
    print("      • Verb types (motion verbs: fly, drive, run)")
    print("      • Noun categories (living vs. non-living)")
    print("      • Context words (environment: sky, highway, park)")

print("\n2️⃣  KNN CLASSIFICATION:")
print("   📌 KNN follows K-MEANS clustering, not our original labels!")
print("   📌 This is by design - KNN learned from K-Means results")
if knn_accuracy >= 0.66:
    print("   ✅ Successfully classified new sentences")
    print("   ✅ Consistent with K-Means learned patterns")
else:
    print("   ⚠️  Some misclassifications occurred")
    print("   ⚠️  Test sentences may have ambiguous features")

print("\n3️⃣  KEY INSIGHTS:")
print("   💡 Unsupervised learning (K-Means) finds hidden patterns")
print("   💡 These patterns may differ from human categorization")
print("   💡 Supervised learning (KNN) then learns these patterns")
print("   💡 Word embeddings capture semantic similarity")

print("\n" + "="*70)
print("✨ ANALYSIS COMPLETE!")
print("="*70)
print("\n📊 Check the generated visualizations for detailed insights!")
print("📄 README.md and PRD.md contain full documentation")

plt.show()