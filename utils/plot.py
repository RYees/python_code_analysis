
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_elbow_point(x, y):
    """
    Find the elbow point in the curve using the second derivative method.
    """
    dy = [y[i] - y[i-1] for i in range(1, len(y))]
    ddy = [dy[i] - dy[i-1] for i in range(1, len(dy))]
    elbow_index = ddy.index(max(ddy)) + 1
    return x[elbow_index]

def silhouette_to_find_optimal_k(job_embeddings):    
    k_values = range(2, 11)
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_label = kmeans.fit_predict(job_embeddings)
        silhouette_avg = silhouette_score(job_embeddings, cluster_label)
        silhouette_scores.append(silhouette_avg)

    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K (using BERT)')
    plt.xticks(k_values)
    plt.show()
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k

def calculate_matched_percentages(level_df, cluster_labels, decimal_places=2):
    percentages = {}
    total_samples = len(cluster_labels)
    
    for label in np.unique(cluster_labels):
        matched_labels = level_df.loc[cluster_labels == label, 'label']
        unique, counts = np.unique(matched_labels, return_counts=True)
        
        percentages[label] = {}
        
        for txt, count in zip(unique, counts):
            percentage = count / total_samples * 100
            rounded_percentage = round(percentage, decimal_places)
            percentages[label][txt] = rounded_percentage
    
    return percentages

def visualize_clusters(level_embeddings, cluster_labels, level_df):
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(level_embeddings)

    percentages = calculate_matched_percentages(level_df, cluster_labels)

    plt.figure(figsize=(8, 6))
    for label in np.unique(cluster_labels):
        if label in percentages:
            label_text = '\n'.join([f'{key}: {value}%' for key, value in percentages[label].items()])
            plt.scatter(embeddings_2d[cluster_labels == label, 0], embeddings_2d[cluster_labels == label, 1],
                            label=label_text)
        else:
            label_text = f'Cluster {label}'
            plt.scatter(embeddings_2d[cluster_labels == label, 0], embeddings_2d[cluster_labels == label, 1],
                            label=label_text)

    plt.title('Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.xticks(rotation='vertical')    
    plt.show()

def visualize_clusters_3d(job_embeddings, cluster_labels, level_df):
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(job_embeddings)

    percentages = calculate_matched_percentages(level_df, cluster_labels)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for label in np.unique(cluster_labels):      
        if label in percentages:
            label_text = '\n'.join([f'{key}: {value}%' for key, value in percentages[label].items()])
            ax.scatter(embeddings_3d[cluster_labels == label, 0], embeddings_3d[cluster_labels == label, 1], embeddings_3d[cluster_labels == label, 2],
                    label=label_text)
        else:
            label_text = f'Cluster {label}'
            ax.scatter(embeddings_3d[cluster_labels == label, 0], embeddings_3d[cluster_labels == label, 1], embeddings_3d[cluster_labels == label, 2],
                    label=label_text)
    ax.set_title('Clustering Visualization (3D)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()

    plt.show()
