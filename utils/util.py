import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import dotenv_values

env_vars = dotenv_values('.env')
openai_api_key  = env_vars.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

def embedding(samples):
    # text-embedding-3-small text-embedding-ada-002
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedded_documents = embed_model.embed_documents(samples)
    return embedded_documents

def silhouette_to_find_optimal_k(level_embeddings):    
    k_values = range(2, 11)
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_label = kmeans.fit_predict(level_embeddings)
        silhouette_avg = silhouette_score(level_embeddings, cluster_label)
        silhouette_scores.append(silhouette_avg)
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k

def kmeans_clustering(optimal_k, embedded_documents):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedded_documents)
    # centroids = kmeans.cluster_centers_
    return cluster_labels, embedded_documents

def Fetch_Each_clustering_Content_By_File(optimal_k, texts, cluster_labels):
    for cluster_label in range(optimal_k):
        cluster_texts = np.array(texts)[np.array(cluster_labels) == cluster_label]
        save_path = f'../txt/w7/cluster_{cluster_label}_texts.txt'

        with open(save_path, 'w') as file:
            for text in cluster_texts:
                file.write(text + '\n')
    
    return cluster_labels

def Fetch_Each_clustering_Content_By_ListofArray(optimal_k, texts, cluster_labels):
    clusters = [[] for _ in range(optimal_k)]  

    for text, label in zip(texts, cluster_labels):
        clusters[label].append(text)  
    
    return clusters
