import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import pdfplumber
import pandas as pd

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def extract_tables_with_pdfplumber(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                table = page.extract_table()
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    return tables

def extract_images_and_text_from_pdf(pdf_docs):
    images_with_context = []
    for pdf in pdf_docs:
        pdf_document = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            page_text = page.get_text("text")
            text_block = page_text.split('\n')

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                image_context = ""
                image_position = len(text_block) // 2
                start = max(0, image_position - 2)
                end = min(len(text_block), image_position + 2)
                image_context = " ".join(text_block[start:end])

                images_with_context.append({
                    "info": f"Page {page_num + 1}, Image {img_index + 1}",
                    "image": image,
                    "base64": img_base64,
                    "context": image_context
                })

    return images_with_context

def filter_relevant_images(images, question):
    relevant_images = []
    question_keywords = question.lower().split()

    for img in images:
        img_context = img["context"].lower()
        if any(keyword in img_context for keyword in question_keywords):
            relevant_images.append(img)

    return relevant_images

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    images = extract_images_and_text_from_pdf(pdf_docs)
    tables = extract_tables_with_pdfplumber(pdf_docs)

    relevant_images = []
    if any(keyword in user_question.lower() for keyword in ["image", "picture", "visual"]):
        relevant_images = filter_relevant_images(images, user_question)

    return response["output_text"], relevant_images, tables

def display_images_in_streamlit(images):
    for img in images:
        st.image(img["image"], use_container_width=False)

def display_tables_in_streamlit(tables):
    for i, table in enumerate(tables):
        st.write(f"**Table {i+1}:**")
        if isinstance(table, pd.DataFrame):
            st.dataframe(table)
        else:
            st.write(table)

def main():
    st.set_page_config("Chat PDF with Images and Tables")

    with open("style.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.header("PDF Chat with Images and Tables")

    if "history" not in st.session_state:
        st.session_state.history = []

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    if pdf_docs:
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            response_text, images, tables = user_input(user_question, pdf_docs)

            st.session_state.history.append({
                "question": user_question,
                "answer": response_text,
                "images": images,
                "tables": tables
            })

            st.write("**Answer:**")
            st.write(response_text)

            if images:
                st.write("**Images:**")
                display_images_in_streamlit(images)

            if tables:
                st.write("**Tables:**")
                display_tables_in_streamlit(tables)

        st.sidebar.header("Conversation History")
        for entry in st.session_state.history:
            st.sidebar.write(f"**Q:** {entry['question']}")
            st.sidebar.write(f"**A:** {entry['answer']}")

            if entry["images"]:
                st.sidebar.write("**Images in Answer:**")
                for img in entry["images"]:
                    st.sidebar.image(img["image"], caption=img["info"], use_container_width=False)

            if entry["tables"]:
                st.sidebar.write("**Tables in Answer:**")
                for table in entry["tables"]:
                    if isinstance(table, pd.DataFrame):
                        st.sidebar.dataframe(table)
                    else:
                        st.sidebar.write(table)

if __name__ == "__main__":
    main()
