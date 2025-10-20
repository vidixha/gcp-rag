import os
import json
from google.cloud import aiplatform
from google.cloud import storage
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

GCP_PROJECT_ID = "your-gcp-project-id"
GCP_REGION = "your-region" 
VECTOR_SEARCH_INDEX_ID = "your-index-id"
VECTOR_SEARCH_ENDPOINT_ID = "your-index-endpoint-id"

storage_client = storage.Client()
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
embeddings_service = None
vector_search_index_endpoint = None

def process_document(cloudevent):
    global embeddings_service, vector_search_index_endpoint

    if embeddings_service is None:
        embeddings_service = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    if vector_search_index_endpoint is None:
        vector_search_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=VECTOR_SEARCH_ENDPOINT_ID)

    event_data = json.loads(cloudevent.data)
    bucket_name = event_data['bucket']
    file_name = event_data['name']
    
    print(f"Processing started for file: {file_name} in bucket: {bucket_name}")

    # Download file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    temp_file_path = f"/tmp/{os.path.basename(file_name)}"
    blob.download_to_filename(temp_file_path)

    try:
        # Load document
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext == '.pdf': loader = PyPDFLoader(temp_file_path)
        elif file_ext == '.docx': loader = UnstructuredWordDocumentLoader(temp_file_path)
        elif file_ext == '.pptx': loader = UnstructuredPowerPointLoader(temp_file_path)
        else:
            print(f"Unsupported file type: {file_ext}"); return
        documents = loader.load()
        
        # Chunk and embed
        for doc in documents: doc.metadata["source_document"] = file_name
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embeddings_service.embed_documents(chunk_texts)
        
        # Prepare datapoints for Vector Search
        datapoints = []
        for i, text in enumerate(chunk_texts):
            datapoints.append({
                "datapoint_id": f"{file_name}-{i}",
                "feature_vector": embeddings[i],
                "restricts": [
                    {"namespace": "source", "allow_list": [file_name]},
                    {"namespace": "text", "allow_list": [text]}
                ]
            })
        
        # Upsert to Vector Search
        vector_search_index_endpoint.upsert_datapoints(index_name=VECTOR_SEARCH_INDEX_ID, datapoints=datapoints)
        print(f"Successfully processed and stored {len(datapoints)} chunks for {file_name}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        os.remove(temp_file_path)
