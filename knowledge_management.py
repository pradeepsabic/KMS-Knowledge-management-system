import os
from extraction.extract import extract_text_from_file
import spacy
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import uuid  # To generate unique IDs
import json 

def get_files_from_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

# Example usage
directory_path = 'Docs'
files = get_files_from_directory(directory_path)
print("Files to process:", files)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').encode

def classify_text(text):
    doc = nlp(text)
    org_count = len([ent for ent in doc.ents if ent.label_ == "ORG"])
    person_count = len([ent for ent in doc.ents if ent.label_ == "PERSON"])
    return {"organizations": org_count, "people": person_count}

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Initialize Chroma DB client
client = chromadb.Client()

# Create or connect to a collection for your documents
collection = client.create_collection("knowledge_management")

def store_document_metadata(file_name, text, classification, entities, embedding_model):
    try:
        # Generate unique embedding and document ID
        embedding = embedding_model(text)
        document_id = str(uuid.uuid4())  # Create a unique ID for the document

        # Convert classification (a dict) to a string (JSON format)
        document = {
            "file_name": file_name,
            "text": text,
            "classification": json.dumps(classification),  # Convert dict to JSON string
            "entities": json.dumps(entities)
        }

        # Add document and embeddings to the Chroma collection
        collection.add(
            documents=[text],
            metadatas=[document],
            embeddings=[embedding],
            ids=[document_id]  # Provide the unique ID here
        )
        print(f"Document {file_name} added to Chroma DB with ID: {document_id}")
    except Exception as e:
        print(f"Error storing document {file_name}: {e}")

# Process each file
for file in files:
    try:
        text = extract_text_from_file(file)
        
        if text:
            print(f"Extracted text from {file}:\n{text[:200]}...")
            
            classification = classify_text(text)
            entities = extract_entities(text)
            
            store_document_metadata(file, text, classification, entities, embedding_model)
            
        else:
            print(f"Failed to extract text from {file}.")
            
    except Exception as e:
        print(f"Error processing {file}: {e}")
def search_documents_with_embedding(query, embedding_model, collection):
    # Generate an embedding for the query
    query_embedding = embedding_model(query)
    
    # Perform similarity search in Chroma DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Number of documents to retrieve
    )
    
    return results

# Example search
query = "machine learning techniques"
search_results = search_documents_with_embedding(query, embedding_model, collection)
print("Search Results:", search_results)

def search_documents_by_metadata(collection, metadata_filter):
    results = collection.query(
        query_texts=[""],
        n_results=5,  # Number of documents to retrieve
        where=metadata_filter  # Filter based on metadata
    )
    return results

# Example usage
metadata_filter = {"classification.organizations": {"$gte": 2}}  # Retrieve documents with 2 or more organizations
metadata_results = search_documents_by_metadata(collection, metadata_filter)
print("Metadata Search Results:", metadata_results)