import streamlit as st
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIG ---
st.set_page_config(page_title="CrediTrust AI Analyst", page_icon="🛡️", layout="wide")

st.title("CrediTrust Complaint Analyst")
st.markdown("### Intelligent RAG Dashboard for Product & Compliance Teams")

# --- MODEL LOADING ---
@st.cache_resource
def load_system():
    # 1. Load Embeddings
    local_embed_path = "models/all-MiniLM-L6-v2/"
    embeddings = HuggingFaceEmbeddings(model_name=local_embed_path)
    
    # 2. Load Vector Store
    vector_db = FAISS.load_local(
        "vector_store/complaints_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 3. Load Local LLM
    local_llm_path = "models/smollm-135m/"
    tokenizer = AutoTokenizer.from_pretrained(local_llm_path)
    model = AutoModelForCausalLM.from_pretrained(local_llm_path)
    
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True
    )
    
    return vector_db, gen_pipeline

with st.spinner("Initializing AI Models... Please wait."):
    vector_db, generator = load_system()

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Question")
    query = st.text_input("e.g., 'What are the main issues with Credit Card fees?'")
    ask_button = st.button("Analyze Complaints")

    if ask_button and query:
        with st.spinner("Retrieving evidence and generating analysis..."):
            # 1. Retrieval
            docs = vector_db.similarity_search(query, k=3)
            context_text = "\n".join([d.page_content for d in docs])
            
            # 2. Prompting
            prompt = f"<|im_start|>system\nYou are a financial analyst. Answer the question using ONLY the context provided.<|im_end|>\n<|im_start|>user\nContext: {context_text}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
            
            # 3. Generation
            output = generator(prompt)
            answer = output[0]['generated_text'].split("<|im_start|>assistant\n")[-1]
            
            st.markdown("---")
            st.markdown("#### 🤖 AI Analyst Response")
            st.write(answer.strip())

with col2:
    st.subheader("📚 Evidence (Source Chunks)")
    if ask_button and query:
        for i, doc in enumerate(docs):
            with st.expander(f"Source {i+1} (ID: {doc.metadata['complaint_id']})"):
                st.info(f"Product: {doc.metadata['product']}")
                st.write(doc.page_content)
    else:
        st.write("Retrieved snippets will appear here after analysis.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info(" No data leaves the CrediTrust network.")
if st.sidebar.button("Clear Conversation"):
    st.rerun()