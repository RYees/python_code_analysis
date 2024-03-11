
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

def find_elbow_point(x, y):
    """
    Find the elbow point in the curve using the second derivative method.
    """
    dy = [y[i] - y[i-1] for i in range(1, len(y))]
    ddy = [dy[i] - dy[i-1] for i in range(1, len(dy))]
    elbow_index = ddy.index(max(ddy)) + 1
    return x[elbow_index]

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

#  UMAP VISUALIZATION
def visualize_with_umap_2d(level_embeddings, cluster_labels, level_df):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    embeddings_2d = umap_2d.fit_transform(level_embeddings)

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

def visualize_with_umap_3d(level_embeddings, cluster_labels, level_df):
    umap_2d = UMAP(n_components=3, init='random', random_state=0)
    embeddings_3d = umap_2d.fit_transform(level_embeddings)

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


#  TSNE VISUALIZATIOIN
def visualize_with_tsne_2d(level_embeddings, cluster_labels, level_df):
    level_embeddings = np.array(level_embeddings)  
    tsne = TSNE(n_components=2, perplexity=5, random_state=0)  
    embeddings_2d = tsne.fit_transform(level_embeddings)

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
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.xticks(rotation='vertical')    
    plt.show()

def visualize_with_tsne_3d(level_embeddings, cluster_labels, level_df):
    level_embeddings = np.array(level_embeddings)  # Convert to NumPy array
    tsne = TSNE(n_components=3, perplexity=5, random_state=0)  # Adjust perplexity as needed
    embeddings_3d = tsne.fit_transform(level_embeddings)

    percentages = calculate_matched_percentages(level_df, cluster_labels)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(cluster_labels):
        if label in percentages:
            label_text = '\n'.join([f'{key}: {value}%' for key, value in percentages[label].items()])
            ax.scatter(embeddings_3d[cluster_labels == label, 0], embeddings_3d[cluster_labels == label, 1],
                       embeddings_3d[cluster_labels == label, 2], label=label_text)
        else:
            label_text = f'Cluster {label}'
            ax.scatter(embeddings_3d[cluster_labels == label, 0], embeddings_3d[cluster_labels == label, 1],
                       embeddings_3d[cluster_labels == label, 2], label=label_text)

    ax.set_title('Clustering Visualization')
    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')
    ax.legend()
    plt.show()

#  PCA VISUALIZATION
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
