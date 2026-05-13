"""
Pdf reader is the predefined class
Pdfreader is used to read data from pdf file
"""
from pypdf import PdfReader

"""
    Sentence Transformer is the predefined class
    SentenceTransformer is used to implement the embeddings

"""
from sentence_transformers import SentenceTransformer

# used to connect to VectorDB
import chromadb

# OpenAI - used to generate output
from openai import OpenAI


# load the model
model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.Client()
collection = client.create_collection("pdf_data")
# read the pdf file

def read_pdf(pdf_path): ## post -- reactjs -- oracle.pdf
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text+= extracted_text

    return text        

# chunks
def chunk_text(text):
    chunk_size = 500 ## LLM able to handle 500 words that is the reason for this chunk
    chunks = []
    
    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

# embeddings 
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# store in db
def store_in_chromadb(chunks,embeddings):
    collection.add(documents = chunks,
                   embeddings=embeddings.tolist(),
                   ids=[str(i) for i in range(len(chunks))])
    
    return "Data Stored Successfully !!!"

# search

def search_query(question):
    query_embedding = model.encode([question])

    context = collection.query(query_embeddings=query_embedding.tolist(),n_results=2)

    return context

# generate output
def generate_output(question,context):
    openAI_client = OpenAI()
    prompt = f"""

        Answer the question using below context only 

        Context:
        {context}

        Question:
        {question}
        """
    
    response = openAI_client.chat.completions.create(model="gpt-4.1-mini",messages=[{"role":"user","content":prompt}])

    final_answer = response.choices[0].message.content 
    return final_answer