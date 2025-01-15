import pandas as pd
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets (for comparison purpose)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Gaussian Mixture Model (EM Algorithm)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_pred = gmm.fit_predict(X_test)

# 2. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_pred = kmeans.fit_predict(X_test)

# 3. Adjusted Rand Index for both methods
ari_gmm = adjusted_rand_score(y_test, gmm_pred)
ari_kmeans = adjusted_rand_score(y_test, kmeans_pred)

# Print the results
print(f"Adjusted Rand Index for Gaussian Mixture Model (EM): {ari_gmm:.4f}")
print(f"Adjusted Rand Index for K-Means: {ari_kmeans:.4f}")

# Comment on clustering quality
if ari_gmm > ari_kmeans:
    print("\nEM (Gaussian Mixture Model) provides better clustering quality.")
else:
    print("\nK-Means provides better clustering quality.")
