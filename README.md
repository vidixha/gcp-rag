# RAG API Assignment

This project implements a decoupled RAG (Retrieval-Augmented Generation) system on Google Cloud Platform.

## Architecture
1.  **API (`rag-api`)**: A FastAPI application with two endpoints:
    * `/upload/`: Uploads a document to a GCS bucket.
    * `/query/`: Performs RAG on the stored documents.
2.  **Processing (`embedding-function`)**: A GCP Cloud Function that triggers on file upload. It reads the document, chunks the text, generates embeddings with Vertex AI, and stores them in Vertex AI Vector Search.

## Setup

1.  **GCP**: Ensure you have a GCP project with the Cloud Functions, Cloud Storage, and Vertex AI APIs enabled. Create a GCS bucket and a Vertex AI Vector Search Index/Endpoint.
2.  **Fill in Details**: Update the placeholder variables (project ID, bucket name, etc.) in both `rag-api/main.py` and `embedding-function/main.py`.
3.  **API**:
    * `cd rag-api`
    * `python3 -m venv venv`
    * `source venv/bin/activate`
    * `pip install -r requirements.txt`
    * `uvicorn main:app --reload`
4.  **Function**:
    * `cd embedding-function`
    * Deploy using the gcloud CLI:
    ```bash
    gcloud functions deploy process-and-embed-document \
    --gen2 \
    --runtime=python311 \
    --region=your-region \
    --source=. \
    --entry-point=process_document \
    --trigger-bucket=your-gcs-bucket-name
    ```
