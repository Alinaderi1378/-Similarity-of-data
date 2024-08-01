import pandas as pd
from sklearn.cluster import KMeans
import tkinter as tk
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
# path excel
file_path = r'C:\Users\Rayan\Desktop\project_uni\Results-AzadKol.xlsx'

# read excel
data = pd.read_excel(file_path)

features = data.iloc[:, 1:] 

dist_matrix = pairwise_distances(features, metric='euclidean')


dist_matrix = np.triu(dist_matrix)

# array
linkage_matrix = linkage(dist_matrix, method='ward')

k = 30

# modelK-Means
kmeans = KMeans(n_clusters=k)

# train
kmeans.fit(features)

# labels
labels = kmeans.labels_

# labels
data['user'] = labels

# matris
plt.imshow(dist_matrix, cmap='hot')
plt.colorbar()
plt.title('Euclidean distance matrix')
plt.show()

# chart
dendrogram(linkage_matrix)
plt.title('Pie chart')
plt.xlabel('USERS')
plt.ylabel('Euclidean distance')
plt.show()

maxSimilarity = np.min(dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)])
minSimilarity = np.max(dist_matrix)

maxSimilarity_indices = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
minSimilarity_indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)


if maxSimilarity_indices[0] == maxSimilarity_indices[1]:
    maxSimilarity_indices = maxSimilarity_indices[0], np.argmin(np.delete(dist_matrix[maxSimilarity_indices[0]], maxSimilarity_indices[0]))
if minSimilarity_indices[0] == minSimilarity_indices[1]:
    minSimilarity_indices = minSimilarity_indices[0], np.argmax(np.delete(dist_matrix[minSimilarity_indices[0]], minSimilarity_indices[0]))

MaxSimilarity_users = [data.index[maxSimilarity_indices[0]], data.index[maxSimilarity_indices[1]]]
minSimilarity_users = [data.index[minSimilarity_indices[0]], data.index[minSimilarity_indices[1]]]
fig, ax = plt.subplots()

plt.title('Similarity Analysis')
ax.text(0.5, 0.5, f"Maximum Similarity: {maxSimilarity:.2f}\nUsers: {MaxSimilarity_users}\n\n"
                  f"Minimum Similarity: {minSimilarity:.2f}\nUsers: {minSimilarity_users}",
        fontsize=14, ha='center', va='center')
ax.axis('off')
plt.show()