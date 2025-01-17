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
import base64  # For encoding images
import tabula  # Optional: for table extraction
import pandas as pd  # Import pandas to handle tables

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract tables from PDFs (using tabula)
def extract_tables_with_tabula(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        tables_from_pdf = tabula.read_pdf(pdf, pages='all', multiple_tables=True)
        
        # Convert each extracted table into pandas DataFrame if necessary
        for table in tables_from_pdf:
            if not isinstance(table, pd.DataFrame):
                table = pd.DataFrame(table)  # Convert list of lists to DataFrame
            tables.append(table)
    return tables

# Function to extract images with their nearby text context from PDFs
def extract_images_and_text_from_pdf(pdf_docs):
    images_with_context = []
    for pdf in pdf_docs:
        pdf_document = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            # Extract text surrounding images
            page_text = page.get_text("text")  # Full text of the page
            text_block = page_text.split('\n')  # Split text into blocks or lines

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                # Get the immediate surrounding text (above and below the image)
                image_context = ""
                image_position = len(text_block) // 2  # Rough position of image in the page text

                # Get text from a small window around the image
                start = max(0, image_position - 2)  # Two lines above
                end = min(len(text_block), image_position + 2)  # Two lines below
                image_context = " ".join(text_block[start:end])

                images_with_context.append({
                    "info": f"Page {page_num + 1}, Image {img_index + 1}",
                    "image": image,
                    "base64": img_base64,
                    "context": image_context  # Add nearby text context
                })

    return images_with_context

# Function to filter relevant images based on the question's relevance to the image's surrounding text
def filter_relevant_images(images, question):
    """
    Filters the images to only show those that are relevant to the user's question.
    The question is matched with the surrounding text context (above and below the image).
    """
    relevant_images = []

    # Keywords (relevant words) from the question
    question_keywords = question.lower().split()  # Case insensitive split into keywords

    for img in images:
        img_context = img["context"].lower()  # Convert context to lowercase for matching
        if any(keyword in img_context for keyword in question_keywords):  # Matching relevant context
            relevant_images.append(img)

    return relevant_images


# Function to save text chunks as embeddings in a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain
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

# Function to handle user queries and responses
def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Extract images and tables
    images = extract_images_and_text_from_pdf(pdf_docs)
    tables = extract_tables_with_tabula(pdf_docs)

    # Filter relevant images based on the user's question and surrounding text
    relevant_images = []
    
    # Show images only if the question asks for them (keywords "image", "visual", "picture")
    if any(keyword in user_question.lower() for keyword in ["image", "picture", "visual"]):
        relevant_images = filter_relevant_images(images, user_question)

    return response["output_text"], relevant_images, tables

# Function to display images with "Zoom/View" and "Download" options
def display_images_in_streamlit(images):
    for img in images:
        st.image(img["image"], use_container_width=False)  # Display only the image without caption


# Function to display tables properly in Streamlit (both for current page and in conversation history)
def display_tables_in_streamlit(tables):
    for i, table in enumerate(tables):
        st.write(f"**Table {i+1}:**")
        # Ensure that tables are displayed as DataFrames
        if isinstance(table, pd.DataFrame):  # Check if table is a DataFrame
            st.dataframe(table)  # This renders the table properly
        else:
            st.write(table)  # In case the table is not in DataFrame format, display as text


# Main Streamlit application
def main():
    st.set_page_config("Chat PDF with Images and Tables")

    # Load custom CSS from a separate file
    with open("style.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.header("PDF Chat with Images and Tables")

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar for file upload
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

            # Add to history
            st.session_state.history.append({
                "question": user_question,
                "answer": response_text,
                "images": images,
                "tables": tables
            })

            # Display answer
            st.write("**Answer:**")
            st.write(response_text)

            # Display images if relevant to the question
            if images:
                st.write("**Images:**")
                display_images_in_streamlit(images)

            # Display tables
            if tables:
                st.write("**Tables:**")
                display_tables_in_streamlit(tables)

        # Show conversation history with tables properly displayed
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
                    # Render table properly in conversation history
                    if isinstance(table, pd.DataFrame):
                        st.sidebar.dataframe(table)  # Proper table rendering
                    else:
                        st.sidebar.write(table)  # If table is not a DataFrame, show as text


if __name__ == "__main__":
    main()
