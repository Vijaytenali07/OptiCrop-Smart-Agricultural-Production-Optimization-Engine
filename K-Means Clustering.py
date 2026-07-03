# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# K-Means Clustering
# ==========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update your dataset path

# ----------------------------------------------------------
# Select Numerical Features
# ----------------------------------------------------------
numerical_features = [
    'Nitrogen',
    'Phosphorus',
    'Potassium',
    'Temperature',
    'Humidity',
    'pH',
    'Rainfall'
]

# Keep only existing columns
numerical_features = [col for col in numerical_features if col in df.columns]

if len(numerical_features) < 2:
    raise ValueError("At least two numerical features are required for clustering.")

X = df[numerical_features]

# ----------------------------------------------------------
# Handle Missing Values
# ----------------------------------------------------------
X = X.fillna(X.mean())

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# Determine Optimal Number of Clusters (Elbow Method)
# ==========================================================

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.grid(True)
plt.show()

# ==========================================================
# Silhouette Score
# ==========================================================

print("\nSilhouette Scores\n")

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print(f"K = {k} --> Silhouette Score = {score:.4f}")

# ==========================================================
# Train Final K-Means Model
# ==========================================================

optimal_k = 3      # Change based on Elbow/Silhouette analysis

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add Cluster Labels
df["Cluster"] = clusters

# ==========================================================
# Cluster Summary
# ==========================================================

print("\nCluster Counts\n")
print(df["Cluster"].value_counts())

print("\nCluster Statistics\n")
print(df.groupby("Cluster")[numerical_features].mean())

# ==========================================================
# Visualize Clusters (First Two Features)
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=X_scaled[:,0],
    y=X_scaled[:,1],
    hue=df["Cluster"],
    palette="Set1",
    s=70
)

plt.title("K-Means Clustering")
plt.xlabel(numerical_features[0])
plt.ylabel(numerical_features[1])

plt.legend(title="Cluster")

plt.show()

# ==========================================================
# Cluster Centers
# ==========================================================

centers = scaler.inverse_transform(kmeans.cluster_centers_)

cluster_centers = pd.DataFrame(
    centers,
    columns=numerical_features
)

print("\nCluster Centers")
print(cluster_centers)

# ==========================================================
# Save Clustered Dataset
# ==========================================================

df.to_csv("OptiCrop_Clustered_Dataset.csv", index=False)

print("\nClustered dataset saved successfully.")

# ==========================================================
# Summary
# ==========================================================

print("\n========== K-Means Clustering Completed ==========")
print(f"Number of Clusters : {optimal_k}")
print(f"Total Samples      : {len(df)}")