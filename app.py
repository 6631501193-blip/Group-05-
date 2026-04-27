import streamlit as st
import torch, os, gc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# --- GPU MEMORY MANAGEMENT ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- INITIALIZE SYSTEM ---
@st.cache_resource
def initialize_system():
    # Update these paths to where your PDFs are stored in Colab
    file_paths = ["/content/sample_data/MLII TOEIC Regis_eng.pdf", "/content/sample_data/MLII TOEIC Regis-TH.pdf"]
    documents = []
    for path in file_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
    
    if not documents:
        st.error("No documents found. Please upload PDFs to /content/sample_data/")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    model_id = "openthaigpt/openthaigpt-1.0.0-7b-chat"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    template = """You are the MLII TOEIC Guide. Respond strictly based on Context.
    If the user says Hi, respond with "Hello! I am your MLII TOEIC Assistant. How can I help you today?".
    Context: {context}
    Question: {question}
    Answer:"""
    
    QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), chain_type_kwargs={"prompt": QA_PROMPT})

# --- UI ---
st.title("MLII TOEIC Assistant")
qa_chain = initialize_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about TOEIC rules:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        clear_vram()
        if prompt.lower().strip() in ["hi", "hello"]:
            res = "Hello! I am your MLII TOEIC Assistant. How can I help you?"
        else:
            res = qa_chain.invoke(prompt)["result"] if qa_chain else "System not initialized."
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})
