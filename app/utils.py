from langchain_community.document_loaders import UnstructuredPDFLoader
import requests
from langchain_openai import ChatOpenAI
from IPython.display import display, Markdown, Image, HTML
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
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from io import BytesIO
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import redis
import streamlit as st

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": 
                                     f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """
    img_base64_list = []
    image_summaries = []
    
    prompt = """You are an assistant tasked with summarizing images for retrieval.
                Remember these images could potentially contain graphs, charts or 
                tables also.
                These summaries will be embedded and used to retrieve the raw image 
                for question answering.
                Give a detailed summary of the image that is well optimized for 
                retrieval.
                Do not add additional words like Summary: etc.
             """
    
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    return img_base64_list, image_summaries

def create_multi_vector_retriever(
    docstore, vectorstore, text_summaries, texts, table_summaries, tables, 
    image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )
    
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
    #add texts, tables, and images
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    
    if image_summaries:
        add_documents(retriever, image_summaries, images)
    return retriever

def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    img_data = base64.b64decode(img_base64)
    img_buffer = BytesIO(img_data)
    img = Image.open(img_buffer)
    display(img)

def plt_img_base64_streamlit(img_base64):
    """Display base64 encoded string as image in Streamlit."""
    img_data = base64.b64decode(img_base64)
    img_buffer = BytesIO(img_data)
    img = Image.open(img_buffer)
    st.image(img)


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8] 
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts (with tables)
    """
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content.decode('utf-8')
        else:
            doc = doc.decode('utf-8')
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def multimodal_prompt_function(data_dict):
    """
    Create a multimodal prompt with both text and image context.
    This function formats the provided context from data_dict, which contains
    text, tables, and base64-encoded images. It joins the text (with table) portions
    and prepares the image(s) in a base64-encoded format to be included in a 
    message.
    The formatted text and images (context) along with the user question are used to
    construct a prompt for GPT-4o mini
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    
    text_message = {
        "type": "text",
        "text": (
            f"""You are an analyst tasked with understanding detailed information 
                and trends from text documents,
                data tables, and charts and graphs in images.
                You will be given context information below which will be a mix of 
                text, tables, and images usually of charts or graphs.
                Use this information to provide answers related to the user 
                question.
                Do not make up answers, use the provided context documents below and 
                answer the question to the best of your ability.
                
                User question:
                {data_dict['question']}
                
                Context documents:
                {formatted_texts}
                
                Answer:
            """
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multimodal_rag_qa(query, multimodal_rag_w_sources):
    response = multimodal_rag_w_sources.invoke({'input': query})
    
    st.markdown('==' * 50)
    st.markdown("### Answer:")
    st.markdown(response['answer']) 

    st.markdown('--' * 50)
    st.markdown("### Sources:")
    i = 1
    img_sources = list(set(response['context']['images'])) 
    for img in img_sources:
        if looks_like_base64(img) and is_image_data(img):
            st.markdown(f"##### Source {i}:")
            i=i+1
            plt_img_base64_streamlit(img)
            st.markdown('-' * 50)

    text_sources = list(set(response['context']['texts'])) 
    for text in text_sources:
        st.markdown(f"##### Source {i}:")
        i=i+1
        st.markdown(text) 
        st.write('')  
        st.markdown('-' * 50)

    st.markdown('==' * 50)
