# Playing with DBSCAN

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# Create new dataset of half-moon shaped structures
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.show()


# First try to classify with k_means
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

km = KMeans(n_clusters=2, random_state=0)

y_km = km.fit_predict(X)

ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], 
            c='lightblue', edgecolor='black',
            marker='o', s=40, 
            label='cluster 1')

ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], 
            c='red', edgecolor='black',
            marker='s', s=40, 
            label='cluster 2')

ax1.set_title('K-means clustering')


# Next work with AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')

y_ac = ac.fit_predict(X)

ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], 
            c='lightblue', edgecolor='black',
            marker='o', s=40, 
            label='cluster 1')

ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1],
            c='red', edgecolor='black',
            marker='s', s=40,
            label='cluster 2')

ax2.set_title('Agglomerative clustering')

plt.legend()
plt.show()



## Now try with DBSCAN
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')

y_db = db.fit_predict(X)

plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], 
            c='lightblue', edgecolor='black',
            marker='o', s=40, 
            label='cluster 1')

plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c='red', edgecolor='black',
            marker='s', s=40,
            label='cluster 2')

plt.legend()
plt.show()
