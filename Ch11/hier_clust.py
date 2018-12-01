### Complete Linkage Hierarchical Clustering

# Generate some random sample data
import pandas as pd
import numpy as np

np.random.seed(123)

variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10

df = pd.DataFrame(X, columns=variables, index=labels)
print()
print(df)
print()

# Calculate distance matrix
from scipy.spatial.distance import pdist, squareform

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels, index=labels)

print()
print(row_dist)
print()


# Apply complete linkage agglomeration to clusters
# Returns linkage matrix
from scipy.cluster.hierarchy import linkage

# incorrect approach (documentation is misleading):
# row_clusters = linkage(row_dist, method='complete', metric='euclidean'

## instead:
# using the condensed matrix 
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete', metric='euclidean')

# or the complete input sample matrix
# row_clusters = linkage(df.values, method='complete', metric='euclidean')


# Turn clustering results into a dataframe
print()
print(pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
             index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])]))


# Visualize results as dendrogram
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# uncomment code below to make dendrogram black
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters, labels=labels) #, make dendrogram black (part 2/2) # color_threshold=np.inf

plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()





## Attaching dendrograms to a heat map

# 1. Create a figure, define positions, width, and height, 
#    and rotate dendrogram 90 degrees counter clockwise
fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

# 2. Reorder data in initial datafram according to the clustering labels
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

# 3. Construct heat map from reordered DataFrame and position next to dendrogram
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

# 4. Remove axis ticks and spines; also add color bar and
#    assign feature and sample names to the x and y axis 
#    tick labels
axd.set_xticks([])
axd.set_yticks([])

for i in axd.spines.values():
    i.set_visible(False)

fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()




# Agglomerative Clustering with SciPy
from sklearn.cluster import AgglomerativeClustering

# n_clusters prunes the tree
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')

labels = ac.fit_predict(X)

# First and fifth sample assigned to one cluster (label1)
# And 2nd and 3rd assigned to label0
# The 4th data point received it's own cluster
print()
print('Cluster labels (n_clusters=3): %s' % labels)
print()

# Rerunning with n_clusters=2
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)

# ID3 should be assigned to the same as 0 and 4 here
print()
print('Cluster labels (n_clusters=2): %s' % labels)
print()

