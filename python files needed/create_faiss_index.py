import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, CSVLoader # Added CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


SOURCE_DOCS_DIR = "/home/spwifi/source_docs"


FAISS_SAVE_PATH = "/home/spwifi/co-lab-files"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_and_save_langchain_faiss_index():
    print("Starting FAISS index creation and saving process...")
    documents = []
    if not os.path.exists(SOURCE_DOCS_DIR):
        print(f"Error: Source documents directory '{SOURCE_DOCS_DIR}' does not exist.")
        print("Please create this directory and place your original PDF/TXT/CSV files inside it.")
        return

    print(f"Loading documents from '{SOURCE_DOCS_DIR}'...")
    try:
        pdf_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())

        txt_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())

        csv_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.csv", loader_cls=CSVLoader)
        documents.extend(csv_loader.load())

    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    if not documents:
        print(f"No documents found in '{SOURCE_DOCS_DIR}'. Please ensure your source files are there.")
        print("Exiting ingestion script.")
        return

    print(f"Loaded {len(documents)} raw documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} text chunks.")

    print(f"Initializing embedding model: '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model initialized.")


    print("Creating LangChain FAISS vector store from document chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"LangChain FAISS vector store created with {len(vectorstore.docstore._dict)} documents in docstore.")

    os.makedirs(FAISS_SAVE_PATH, exist_ok=True) 
    print(f"Saving LangChain FAISS vector store to '{FAISS_SAVE_PATH}'...")
    vectorstore.save_local(FAISS_SAVE_PATH)
    print("FAISS index and docstore saved successfully (both index.faiss and index.pkl are compatible).")

if __name__ == "__main__":
    create_and_save_langchain_faiss_index()
