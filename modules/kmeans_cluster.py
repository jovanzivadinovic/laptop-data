from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.model_selection import train_test_split


class KMeansCluster:
    def find_optimal_k(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        return wcss
    
    def kmeans(self, X, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        labels = kmeans.fit_predict(X_scaled)
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels) 
        plt.tight_layout()
        plt.show()

    
    def calculate_optimal_k(self, wcss):
        kl = KneeLocator(range(1,11), wcss, curve="convex", direction="decreasing")
        return kl.elbow