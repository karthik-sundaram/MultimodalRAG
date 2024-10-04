from langchain_community.document_loaders import UnstructuredPDFLoader
import requests
from langchain_openai import ChatOpenAI
import os
import base64
from langchain_core.messages import HumanMessage
import re
import htmltabletomd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_chroma import Chroma
from langchain_core.documents import Document
from PIL import Image
from io import BytesIO
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import redis
import streamlit as st
import chromadb
from chromadb.api.client import SharedSystemClient


from utils import (
    encode_image,
    image_summarize,
    generate_img_summaries,
    create_multi_vector_retriever,
    plt_img_base64,
    looks_like_base64,
    is_image_data,
    split_image_text_types,
    multimodal_prompt_function,
    multimodal_rag_qa,
)



def main():

    st.title("Multimodal Retrieval-Augmented Generation App")

    OPENAI_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
    if OPENAI_KEY:
        os.environ['OPENAI_API_KEY'] = OPENAI_KEY
    else:
        st.stop()

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("data/uploaded_documents/uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully.")

        with st.spinner("Processing the PDF..."):
            loader = UnstructuredPDFLoader(
                file_path="data/uploaded_documents/uploaded_document.pdf",
                strategy='hi_res',
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=4000,
                combine_text_under_n_chars=2000,
                mode='elements',
                image_output_dir_path='./figures'
            )
            data = loader.load()
        
        img_folder = './figures'
        img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]

        docs = []
        tables = []
        for doc in data:
            if doc.metadata['category'] == 'Table':
                tables.append(doc)
            elif doc.metadata['category'] == 'CompositeElement':
                docs.append(doc)
        for table in tables:
            table.page_content = htmltabletomd.convert_table(table.metadata['text_as_html'])

        st.write(f"##### There are : {len(docs)} Text chunks , {len(tables)} Tables and {len(img_files)} Images in your pdf document")

        chatgpt = ChatOpenAI(model_name='gpt-4o-mini', n=1 , temperature=0)


        # Prompt
        prompt_text = """
        You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
        These summaries will be embedded and used to retrieve the raw text or table elements
        Give a detailed summary of the table or text below that is well optimized for retrieval.
        For any tables also add in a one line description of what the table is about besides the summary.
        Do not add additional words like Summary: etc.
        Table or text chunk:
        {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        summarize_chain = (
                            {"element": RunnablePassthrough()}
                            |
                            prompt
                            |
                            chatgpt
                            |
                            StrOutputParser() 
        )

        text_summaries = []
        table_summaries = []

        text_docs = [doc.page_content for doc in docs]
        table_docs = [table.page_content for table in tables]

        text_summaries = summarize_chain.batch(text_docs, {"max_concurrency": 5})
        table_summaries = summarize_chain.batch(table_docs, {"max_concurrency": 5})

        IMG_PATH = './figures'
        imgs_base64, image_summaries = generate_img_summaries(IMG_PATH) 

        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

        #chroma setup
        SharedSystemClient.clear_system_cache()
        client = chromadb.EphemeralClient()
        chroma_db = Chroma(
            collection_name="mm_rag",
            embedding_function=openai_embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            client=client
        )

        #redis setup
        client = get_client('redis://localhost:6379')
        redis_store = RedisStore(client=client)


        #retriever
        retriever_multi_vector = create_multi_vector_retriever(
            redis_store,  chroma_db,
            text_summaries, text_docs,
            table_summaries, table_docs,
            image_summaries, imgs_base64,
        )

        #rag chain
        multimodal_rag = (
                {
                    "context": itemgetter('context'),
                    "question": itemgetter('input'),
                }
                    |
                RunnableLambda(multimodal_prompt_function)
                    |
                chatgpt
                    |
                StrOutputParser()
        )

        #pass input query to retriever and get context document elements
        retrieve_docs = (itemgetter('input')
                            |
                        retriever_multi_vector
                            |
                        RunnableLambda(split_image_text_types))

        multimodal_rag_w_sources = (RunnablePassthrough.assign(context=retrieve_docs)
                                                    .assign(answer=multimodal_rag)
        )


        st.success("PDF processing complete.")

        # input
        query = st.text_input("Enter your question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                multimodal_rag_qa(query,multimodal_rag_w_sources)



if __name__ == "__main__":
    main()
