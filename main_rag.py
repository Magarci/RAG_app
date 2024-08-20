import streamlit as st
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import getpass
import os

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)





template = """
    You are a RAG application. Your response should use the added documents. The question is: {draft}
"""

# prompttemplate
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(template)
    ],
    input_variables={"draft"},

)

#LLM and key loading function
def load_LLM(groq_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama3-70b-8192", 
    temperature=0.7)
    return llm


#Page title and header
st.set_page_config(page_title="RAG app")
st.header("Aplicacion RAG para analizar cosas")



#Input Groq API Key
st.markdown("## Introduzca su clave API de ChatGroq")

def get_groq_api_key():
    input_text = st.text_input(label="ChatGroq API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

groq_api_key = get_groq_api_key()


# Input
st.markdown("## Introduzca su pregunta")

def get_draft():
    draft_text = st.text_area(label="Text", label_visibility='collapsed', placeholder="Your Text...", key="draft_input")
    return draft_text

draft_input = get_draft()

if len(draft_input.split(" ")) > 700:
    st.write("Por favor, introduzca un texto más corto. La longitud máxima es de 700 palabras.")
    st.stop()

    
# Output
st.markdown("### Su respuesta:")

if draft_input:
    if not groq_api_key:
        st.warning('Please insert groq API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
        st.stop()

    llm = load_LLM(groq_api_key = groq_api_key)

    prompt_with_draft = prompt.format_prompt(
        draft=draft_input
    )

    improved_redaction = llm.invoke(prompt_with_draft)
    improved_redaction_content = improved_redaction.content
    st.write(improved_redaction_content)