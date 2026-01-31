import streamlit as st
import tempfile
from rag_sql import extract_pdf_text, build_vector_store, generate_sql

st.set_page_config(page_title="PDF → SQL Generator")
st.title("📄 PDF Schema → SQL Generator")

uploaded_pdf = st.file_uploader("Upload PDF Schema", type="pdf")

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    with st.spinner("Reading schema..."):
        text = extract_pdf_text(pdf_path)
        chunks, model, index = build_vector_store(text)

    st.success("Schema loaded")

    query = st.text_input("Ask a SQL question")

    if query:
        sql = generate_sql(query, chunks, model, index)
        st.subheader("Generated SQL")
        st.code(sql, language="sql")
