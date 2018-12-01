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



