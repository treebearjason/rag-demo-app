import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import tempfile
from config import DATA_DIR
import os
from doc_handler import DocumentHander
from query_service import QueryService



if __name__ == "__main__":
    doc_handler = DocumentHander()
    query_service = QueryService()
    

    ## Side bar
    with st.sidebar:
        st.set_page_config(page_title="RAG Q/A Demo")
        
        uploaded_file = st.file_uploader("**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False)

        col1, col2 = st.columns(2)
        with col1:
            process = st.button("⚡️ Process",)
        with col2:
            is_reset = st.checkbox("Vector DB reset?")

        if uploaded_file and process:
            doc_handler.save_document(uploaded_file)
            doc_handler.process(is_reset)
            st.success("Data added to the vector store!")

    ## Main Page
    st.header("🗣️RAG Q/A Demo")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask",)

    if ask and prompt:
        relevant_text, relevant_text_ids = query_service.search_docs(prompt)
        response = query_service.call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See most relevant documents"):
            st.write(relevant_text_ids)
            st.write(relevant_text)

