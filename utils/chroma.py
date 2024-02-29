#from helper_utils import word_wrap # helper functions from utilities
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from dotenv import dotenv_values
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai

# client = OpenAI(api_key=env_vars.get('OPENAI_API_KEY'),

env_vars = dotenv_values('.env')
api_key=env_vars.get('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=openai.api_key)

def pdf_reader(file_path):
    reader = PdfReader(file_path)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    # print(pdf_texts[0])
    return pdf_texts

def text_splitter_chunks(pdf_texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    # print(character_split_texts[10])
    # print(f"\nTotal chunks: {len(character_split_texts)}")
    return character_split_texts

def sentence_transfomer_textsplitter(character_split_texts):
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)
    # print(token_split_texts[10])
    # print(f"\nTotal chunks: {len(token_split_texts)}")
    return token_split_texts


def embedding(token_split_texts):
    embedding_function = SentenceTransformerEmbeddingFunction()
    #print(embedding_function([token_split_texts[10]]))
    return embedding_function

def connect_with_chromadb(embedding_function, token_split_texts, collection_name):
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)

    ids = [str(i) for i in range(len(token_split_texts))]

    chroma_collection.add(ids=ids, documents=token_split_texts)
    chroma_collection.count()
    return chroma_collection

def vectordb_answer_question(query, chroma_collection):
    # query = "What was the total revenue?"
    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]

    for document in retrieved_documents:
        print(document)
        print('\n')
    return retrieved_documents

def openai_model_answer(query, retrieved_documents, model="gpt-3.5-turbo"):
    information = "\n\n".join(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content
