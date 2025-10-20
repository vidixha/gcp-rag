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


------------------------------------------------------------

## Architecture




### Component Breakdown

* **User**: Interacts with the system via the FastAPI application.
* **FastAPI API (Cloud Run)**: The application's front door.
    * **`/upload`**: Receives documents and immediately places them in a Cloud Storage bucket[cite: 3].
    * **`/query`**: Handles user questions, orchestrates the RAG process, and returns a final answer[cite: 11].
* **Cloud Storage**: Acts as the primary "GCP Data Store" for the original documents[cite: 4]. When a new file arrives, it emits an event that triggers the processing function.
* **Cloud Function**: The core of the processing pipeline, triggered by file uploads[cite: 8]. It reads the document, chunks the text, generates vector embeddings, and upserts them into the vector datastore[cite: 7].
* **Vertex AI Embeddings**: The model that converts text chunks into numerical vector representations.
* **Vertex AI Vector Search**: A specialized database that stores vector embeddings for high-speed similarity search.
* **LLM (Gemini on Vertex AI)**: The Large Language Model that generates the final, human-readable answer based on the context provided by the retriever[cite: 15].

---

### Process Flows

#### 1. Document Upload & Processing Flow

1.  A user sends a document (`.pdf`, `.docx`, `.pptx`) to the `/upload` endpoint of the FastAPI API[cite: 3, 6].
2.  The API places the file directly into the **Cloud Storage** bucket[cite: 4].
3.  The file creation event in the bucket automatically triggers the **Cloud Function**[cite: 8].
4.  The function reads the document, splits it into text chunks, and generates embeddings for each chunk using **Vertex AI Embeddings**[cite: 7].
5.  The function then stores these embeddings and their corresponding text in **Vertex AI Vector Search**[cite: 7].

#### 2. RAG Query Flow

1.  A user sends a question to the `/query` endpoint of the FastAPI API[cite: 11].
2.  The API's retriever embeds the user's question using the **Vertex AI Embeddings** model.
3.  The retriever uses this question embedding to query the **Vertex AI Vector Search** index, finding the most semantically similar document chunks. This step ensures the question is mapped to the right documents[cite: 12].
4.  Vector Search returns the relevant text chunks.
5.  The API sends the original question and the retrieved text chunks to the **LLM (Gemini)**[cite: 15].
6.  The LLM generates an answer based *only* on the provided context, fulfilling the core RAG requirement[cite: 13].
7.  The API returns the final answer and source document information to the user.
