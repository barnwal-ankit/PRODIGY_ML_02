import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) 


try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found. Please download from Kaggle.")
    print("https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
    exit()

-

features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans_elbow = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans_elbow.fit(X_scaled)
    wcss.append(kmeans_elbow.inertia_)


plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(k_range)
plt.grid(True)
# plt.show() 

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)


df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7, legend='full')

# Plot centroids
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled) 
plt.scatter(centers_original[:, 0], centers_original[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroids')

plt.title(f'Customer Segments (K={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
print(f"\n--- K-Means Clustering Results ---")
print(f"Features Used: {features}")
print(f"Optimal K chosen: {optimal_k}")
print("\nCluster distribution:")
print(df['Cluster'].value_counts())
print("\nShowing Cluster Visualization...")
plt.show()
