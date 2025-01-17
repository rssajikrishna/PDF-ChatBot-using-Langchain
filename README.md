# PDF ChatBot Using Langchain

## Overview

This is a Streamlit-based web application that allows users to interact with PDFs. The app extracts text, images, and tables from PDF files, and users can ask questions about the content. The app uses the Google Generative AI API for question answering, LangChain for handling text, and FAISS for storing and retrieving embeddings.

## Features

- **PDF Text Extraction**: Extracts text from PDF documents.
- **Image Extraction**: Extracts images from the PDF along with surrounding text context.
- **Table Extraction**: Extracts tables from PDFs and displays them as DataFrames.
- **Question Answering**: Uses the Google Generative AI API for answering questions based on the extracted content from the PDFs.
- **Display of Images and Tables**: Displays images and tables relevant to the user's query.
- **Conversation History**: Maintains the conversation history within the app.

## Requirements

To run this application, make sure you have the following Python packages installed:

```bash
pip install -r requirements.txt
