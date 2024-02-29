import os, sys
import pandas as pd
import sys

path='../../'
if not path in sys.path:
    sys.path.append(path)
    
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from sklearn.manifold import TSNE
from collections import Counter
from sklearn.metrics import pairwise_distances

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()
#from tika import parser
import tiktoken
import time
from openai import OpenAI
from sklearn.decomposition import PCA
#from gdrive import gsheet
from dotenv import dotenv_values

env_vars = dotenv_values('.env')
openai_api_key  = env_vars.get('OPENAI_API_KEY')
#openai_client = openai.OpenAI(api_key=openai.api_key)

# openai_api_key = config.openai.api_key
client = OpenAI(api_key=openai_api_key)


# def get_job_from_gsheet():
        
#         sid = "1EDF3EiHrneLeLkVOo2YUd9m0EMnshcqd0oqorDZpRq4"
#         gst = gsheet(sheetid=sid,fauth='admin-10ac-service.json') 
        
#         dfg = gst.get_sheet_df('dat')
#         job = dfg.T
        # job_desc_n = job_desc[['title','post_link','description','comment']]
        # job_desc_n = job_desc_n.tail(1500)
        # job = job_desc_n.sample(n=100, random_state=42)

        #return job 
    
def elbow_to_find_optimal_k(job_embeddings):
    k_values = range(2, 11)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(job_embeddings)
        inertias.append(kmeans.inertia_)

    # plt.plot(k_values, inertias, marker='o')
    # plt.xlabel('Number of Clusters (K)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for Optimal K')
    # plt.xticks(k_values)
    # plt.show()

    optimal_k = find_elbow_point(k_values, inertias)
    
    
    return optimal_k
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

    # plt.plot(k_values, silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters (K)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score for Optimal K (using BERT)')
    # plt.xticks(k_values)
    # plt.show()
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k

def kmeans_clustering(optimal_k, embedded_documents):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedded_documents)
    # centroids = kmeans.cluster_centers_
    return cluster_labels, embedded_documents

def compute_gap_statistics(data):
    gaps = []
    max_k = 12
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        within_dispersion = np.log(kmeans.inertia_)
        reference_data = np.random.rand(*data.shape)
        reference_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        reference_kmeans.fit(reference_data)
        reference_dispersion = np.log(reference_kmeans.inertia_)
        gap = reference_dispersion - within_dispersion
        gaps.append(gap)
    optimal_k = np.argmax(gaps) + 1
    
    print("Optimal number of clusters (k):", optimal_k)
    
        
    return optimal_k

def plot_gap_statistics(gaps):
    plt.plot(range(1, len(gaps) + 1), gaps, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistics for Optimal K')
    plt.show()
    

def job_clustering_using_text_embedding(jobs_df):
    client = OpenAI(api_key=openai_api_key)
    job_embeddings = []

    for description in jobs_df['code_content']:
        response = client.embeddings.create(
            input=description,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        job_embeddings.append(embedding)

    job_embeddings = np.array(job_embeddings)
    
    optimal_k = silhouette_to_find_optimal_k(job_embeddings)
    # optimal_k = 2
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(job_embeddings)
    # centroids = kmeans.cluster_centers_
    return cluster_labels, job_embeddings

def job_clustering_using_bert(jobs_df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    job_embeddings = []
    for description in jobs_df['description']:
        inputs = tokenizer(description, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average pooling over token embeddings
        job_embeddings.append(embeddings)

    job_embeddings = np.array(job_embeddings)

    # optimal_k = silhouette_to_find_optimal_k(job_embeddings)
    optimal_k = 2
    cluster_labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(job_embeddings)
    return cluster_labels, job_embeddings

def job_clustering_using_paragraph_embedding(jobs_df):
    client = OpenAI(api_key=openai_api_key)
    job_embeddings = []

    for description in jobs_df['description']:
        paragraphs = description.split('\n')
        
        paragraph_embeddings = []
        
        for paragraph in paragraphs:
            response = client.embeddings.create(
                input=paragraph,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            paragraph_embeddings.append(embedding)
        paragraph_embeddings = np.array(paragraph_embeddings)
        job_embeddings.append(paragraph_embeddings)
        
    average_embeddings = [np.mean(job, axis=0) for job in job_embeddings]
    X = np.array(average_embeddings)

    n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_


    return cluster_labels, X

def visualize_clusters(job_embeddings, cluster_labels,jobs_df):
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(job_embeddings)
    
    percentage = calculate_matched_percentages(jobs_df, cluster_labels)
    
    plt.figure(figsize=(8, 6))
    for label in np.unique(cluster_labels):
        if label == 0:
            plt.scatter(embeddings_2d[cluster_labels == label, 0], embeddings_2d[cluster_labels == label, 1], label=f'Data Engineer, Matched:{percentage[0]}%')
        elif label == 1:
            plt.scatter(embeddings_2d[cluster_labels == label, 0], embeddings_2d[cluster_labels == label, 1], label=f'Software Engineering, Matched:{percentage[1]}%')
        else:
            plt.scatter(embeddings_2d[cluster_labels == label, 0], embeddings_2d[cluster_labels == label, 1], label=f'Cluster {label}')
        
    plt.title('Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    
def calculate_matched_percentages(jobs_df, cluster_labels):
    percentages = {}
    for label in np.unique(cluster_labels):
        matched_labels = jobs_df['manual_cluster'][cluster_labels == label]
        unique, counts = np.unique(matched_labels, return_counts=True)
        percentages[label] = {}
        for txt, count in zip(unique, counts):
            percentage = int(count / len(matched_labels) * 100)
            percentages[label] = percentage
    return percentages


def bar_plot(percentages):
    match_de = percentages[0]
    match_se = percentages[1]
    not_match_de = 100 - match_de
    not_match_se = 100 - match_se 

    labels = ['Data Engineer', 'Software Engineer']
    matched_percentages = [match_de, match_se]
    not_matched_percentages = [not_match_de, not_match_se]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, matched_percentages, width, label='Matched %', color='green')
    rects2 = ax.bar(x + width/2, not_matched_percentages, width, label='Not Matched %', color='red')

    ax.set_ylabel('Percentage')
    ax.set_title('Matched and Not Matched Percentages by Label')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

def generating_ideal_cv(template, job_descriptions):
    # template  = "You are a helpful AI assistant capable of generate a comprehensive CV (curriculum vitae)\
    #     based on the following {job_descriptions}. The CV should highlight qualifications, relevant experience, and skills aligning with each positions requirements.\
    #         the generated cv must have the following key points:\
    #         ### Role : the role should be a generalize role that can represent all the job descriptions specializations. Example if the job descriptions position is \
    #             Sr. or senior Data Engineer, Jr. or Junior Data Engineer, Azure Data Engineer, Data Engineer, Data Expert, Cloud Data Engineer, then take Data Engineer as a Role\
    #                 for the CV. if in addition to this data Engineering roles, it also contains Data Scientist . take Data Engineer and Scientist as a role. do same way for the other roles.\
    #         ### Educational Background: this is the educational degree required by the jobs. Example, Bachelors degree in computer Science, Masters science in data engineering\
    #         ### Experiences: the experience should be the required experiences in a particular specialization. don't include experiences in a specific tool or technique. \
    #             Example experiences in data modeling, don't include this in experiences part just put it in Skill category. if you get such generalized experience , experience in data Engineering or software development put in Experiences category.\
    #         ### Skills: put the skills required by the role in this category.\
    #         ### Domain Knowledge and Attitudes:  Add domain knowledge and attitude needed for the roles. \
    #         Note that to cite each contents of the generated CV to respect job descriptions."
            
    prompt = {
        "role": "user",
        "content": template.format(job_descriptions=job_descriptions)
    }
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of generating representative CVs."},
            prompt
        ]
    )
    
    return response.choices[0].message.content

def job_and_cv_matching(cv, job_desc):
    template='''I want to you to check if the following '{cv}' matches the following {job_desc} and show the match degree as highly matched rank 10,\
    low matched rank as 1 and not matched rank as 0'''
    language_prompt = PromptTemplate(
        input_variables=["cv",'job_desc'],
        template=template,
    )
    language_prompt.format(cv="cv",job_desc='job description')

    chain=LLMChain(llm=llm,prompt=language_prompt)
    result = chain({'cv':cv,'job_desc':job_desc})
    return result['text']