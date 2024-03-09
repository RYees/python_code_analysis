import pandas as pd
import numpy as np
import re
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

def Fetch_Each_clustering_Content(optimal_k, texts, cluster_labels):
    for cluster_label in range(optimal_k):
        cluster_texts = np.array(texts)[np.array(cluster_labels) == cluster_label]
        save_path = f'../txt/w7/cluster_{cluster_label}_texts.txt'

        with open(save_path, 'w') as file:
            for text in cluster_texts:
                file.write(text + '\n')
    
    return cluster_labels
