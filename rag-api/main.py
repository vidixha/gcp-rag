import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from google.cloud import aiplatform
from typing import List

from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

GCP_PROJECT_ID = "your-gcp-project-id"
GCP_REGION = "your-region"  
GCS_BUCKET_NAME = "your-gcs-bucket-name"
VECTOR_SEARCH_INDEX_ID = "your-index-id"
VECTOR_SEARCH_ENDPOINT_ID = "your-index-endpoint-id"

storage_client = storage.Client(project=GCP_PROJECT_ID)
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
llm = VertexAI(model_name="gemini-1.0-pro")
embeddings_service = VertexAIEmbeddings(model_name="textembedding-gecko@003")
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=VECTOR_SEARCH_ENDPOINT_ID)

class VertexAIVectorSearchRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = embeddings_service.embed_query(query)
        search_results = index_endpoint.match(
            queries=[query_embedding], num_neighbors=5, deployed_index_id=VECTOR_SEARCH_INDEX_ID
        )
        relevant_docs = []
        if search_results and search_results[0]:
            for match in search_results[0]:
                text = next((r.allow_list[0] for r in match.datapoint.restricts if r.namespace == 'text'), None)
                source = next((r.allow_list[0] for r in match.datapoint.restricts if r.namespace == 'source'), None)
                if text and source:
                    doc = Document(page_content=text, metadata={"source": source, "score": match.distance})
                    relevant_docs.append(doc)
        return relevant_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

app = FastAPI(title="RAG Document API")

@app.post("/upload/", tags=["1. Document Upload"])
async def upload_document(file: UploadFile = File(...)):
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File format not allowed.")
    
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file.file, content_type=file.content_type)
        return {"filename": file.filename, "gcs_path": f"gs://{GCS_BUCKET_NAME}/{file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloud upload failed: {e}")

class QueryRequest(BaseModel):
    question: str

@app.post("/query/", tags=["2. RAG Query"])
async def perform_rag_query(request: QueryRequest):
    try:
        retriever = VertexAIVectorSearchRetriever()
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>{context}</context>
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = await retrieval_chain.ainvoke({"input": request.question})
        
        source_docs = []
        if "context" in response and response["context"]:
            source_docs = [{"source": doc.metadata.get("source"), "score": doc.metadata.get("score")} for doc in response["context"]]

        return {"answer": response.get("answer"), "source_documents": source_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
