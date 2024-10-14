# MultimodalRAG
  
This project demonstrates a Multimodal RAG system where text, tables, and images are processed and retrieved to answer user queries. Users can upload PDFs containing various data types, and the system efficiently retrieves relevant elements and generates detailed answers.

   ![image](https://github.com/user-attachments/assets/be280f45-9b24-43de-a2f9-8f84d4b44253)
   Source: LangChain Blogs

## Key Features:  
- **PDF Ingestion & Processing**: Extracts text, tables, and images from PDFs using UnstructuredPDFLoader.
- **Summarization**: Text, tables, and image content are summarized by GPT-4o-mini to enable semantic retrieval.
- **Storage & Embedding**:
     **a. Redis**: Stores raw content for quick retrieval.
     **b. Chroma**: Converts summaries into embeddings for vector-based search and retrieval.
- **Multimodal Retrieval**: Using MultiVectorRetriever, the system matches and retrieves a combination of text, tables, and images.
- **RAG Pipeline**: Dynamically retrieves relevant elements and generates accurate responses based on user input.
- **Streamlit UI**: Interactive web interface that allows PDF uploads and questions about the content.

## Architecture Overview:
- **Document Parsing**: PDF documents are parsed into distinct chunks – text, tables, and images.
- **Summarization & Embedding**: Summaries are created using GPT-4o-mini and embedded into Chroma.
- **Redis & Chroma Integration**: Redis stores raw data **(docstore)** while Chroma handles vector embeddings **(vector store)** for optimized search.
- **Multimodal RAG Pipeline**: User queries are passed through a RAG chain that retrieves relevant content types (text, tables, and images) and answers questions.
- **Deployment with Docker**: Hosted on Streamlit, the app runs inside a Docker container with Redis, OpenAI’s API, and Chroma for backend services.

![pic10](https://github.com/user-attachments/assets/902708a0-41bb-43e8-b71d-1b9f40c6d2f3)
![pic11](https://github.com/user-attachments/assets/f8af9996-48dc-4220-9ff0-96680c4fccb1)
![pic12](https://github.com/user-attachments/assets/d275fa7d-71d5-41b2-8e77-e1a313c3e8ce)

## Technology Stack:
- **Redis & Chroma**: For hybrid data storage and vector-based retrieval.
- **GPT-4o-mini**: Used for summarization and language generation.
- **Docker**: For containerized deployment.
- **Streamlit**: Web-based interface for interaction with users.


