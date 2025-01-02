from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansCluster:
    def find_optimal_k(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    
    def kmeans(self, X, n_clusters):
        plt.scatter(X[:, 0], X[:, 1], c=n_clusters)
        plt.show()
    