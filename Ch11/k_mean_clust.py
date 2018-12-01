import numpy as np

### k-means Cclustering

# Generate 150 random datapoints, grouped into 3 regions
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150, n_features=2, centers=3, 
                  cluster_std=0.5, shuffle=True, random_state=0)


import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1],
            c='white', edgecolor='black',
            marker='o', s=50)

plt.grid()
plt.show()


# Run k-means
from sklearn.cluster import KMeans

# Set number of desired clusters to 3, run 10 times, tolerance=0.0001
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            c='lightgreen', edgecolor='black',
            marker='s', s=50, label='cluster 1')

plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], 
            c='orange', edgecolor='black',
            marker='s', s=50, label='cluster 2')

plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], 
            c='lightblue', edgecolor='black',
            marker='s', s=50, label='cluster 3')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='red', edgecolor='black',
           marker='*', s=250, label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.show()



# k-means++ algorithm (just change the initialization)

km_pp = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km_pp.fit_predict(X)

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], 
            c='lightgreen', edgecolor='black',
            marker='s', s=50, label='cluster 1')

plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], 
            c='orange', edgecolor='black',
            marker='s', s=50, label='cluster 2')

plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], 
            c='lightblue', edgecolor='black',
            marker='s', s=50, label='cluster 3')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
           c='red', edgecolor='black',
           marker='*', s=250, label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# Show distortion of both models; should be the same for this dataset
print()
print('k-means Distortion: %.2f' % km.inertia_)
print()
print('k-means++ Distortion: %.2f' % km_pp.inertia_)
print()


# Plot distortion for different values of k
distortions = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='k-means++',
                n_init=10, max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

# Viewing the graph w/ elbow method should yield a value of 3
# for cluster number
plt.plot(range(1, 15), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()



# Evalutate with silhouette method
from matplotlib import cm
from sklearn.metrics import silhouette_samples

y_km = km_pp.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
y_ticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper), 
             c_silhouette_vals,
             height=1.0, edgecolor='none',
             color=color)
    y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

# Same as importing silhouette_scores
silhouette_avg = np.mean(silhouette_vals)

plt.axvline(silhouette_avg, color='red', linestyle='--')

plt.yticks(y_ticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()


# Silhouette plot with bad clustering

# Seed model with bad cluster number
km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=0)

y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            c='lightgreen', edgecolor='black',
            marker='s', s=50, label='cluster 1')

plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            c='lightblue', edgecolor='black',
            marker='s', s=50, label='cluster 2')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='red', edgecolor='black',
           marker='*', s=250, label='centroids')

plt.legend()
plt.grid()
plt.show()


cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
y_ticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()

    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0, edgecolor='none',
             color=color)
    y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

# Same as importing silhouette_scores
silhouette_avg = np.mean(silhouette_vals)

plt.axvline(silhouette_avg, color='red', linestyle='--')

plt.yticks(y_ticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()



