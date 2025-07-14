import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document # Import Document for type hinting and clarity

# --- Configuration ---

# Path to the directory containing your FAISS index files (index.faiss and index.pkl)
FAISS_INDEX_PATH = "/home/spwifi/co-lab-files"

# Name of the embedding model used for creating the FAISS index
# This MUST match the model used during ingestion (e.g., in create_faiss_index.py)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Path to the llama.cpp 'main' executable (or 'llama' if you renamed it)
# IMPORTANT: Verify this path. Use 'find / -name "main" 2>/dev/null' to locate it.
LLAMA_CPP_EXECUTABLE_PATH = "/root/llama.cpp/" # <--- VERIFY THIS PATH

# Path to your GGUF model file (e.g., Mistral-7B-Claude-Chat.Q4_K_M.gguf)
# IMPORTANT: Ensure this path and filename exactly match your model
GGUF_MODEL_PATH = "/root/llama.cpp/build/bin/models/co-lab-files/my_mistral_gguf_model/mistral-7b-claude-chat.Q4_K_M.gguf"
 # Verify this path and filename

# Number of layers to offload to the GPU. Set to 0 for CPU only.
# Adjust based on your GPU memory. Common values are 30-40 for 7B models if you have VRAM.
N_GPU_LAYERS = 0

# Number of documents to retrieve initially from FAISS before score filtering
# It's good practice to retrieve more than TOP_K_DOCUMENTS to allow for filtering
INITIAL_RETRIEVAL_COUNT = 10 # Retrieve more to filter down

# FAISS Similarity Score Threshold
# Documents with a score HIGHER than this threshold will be filtered out.
# Lower scores mean higher similarity (L2 distance). Adjust this value based on your data.
# Start with a value like 0.7-1.0 and fine-tune.
SCORE_THRESHOLD = 1.0 # Example: Only include documents with L2 distance <= 1.0

# Threshold for fallback: if retrieved documents (after score filtering) are less than this,
# fall back to LLM only.
# Set to 1 if you want to fall back if NO documents are found after filtering.
FALLBACK_DOC_THRESHOLD = 1

# --- Prompt Templates ---

# Prompt for RAG (when documents are retrieved)
RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Helpful Answer:"""

# Prompt for direct LLM query (when no relevant documents are found)
LLM_ONLY_PROMPT_TEMPLATE = """Answer the following question directly.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}
Answer:"""

def initialize_rag_system():
    """Initializes the embedding model, FAISS vector store, and LLM."""
    print("Initializing RAG System...")

    # 1. Initialize Embedding Model
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure 'sentence-transformers' and 'huggingface_hub' are installed and up-to-date.")
        return None, None, None

    # 2. Load FAISS Vector Store
    print(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")) or \
       not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.pkl")):
        print(f"Error: FAISS index files (index.faiss and index.pkl) not found in '{FAISS_INDEX_PATH}'.")
        print("Please ensure you have run 'create_faiss_index.py' to generate them correctly.")
        return None, None, None
    try:
        # allow_dangerous_deserialization=True is needed for loading pickled objects
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("This might be due to an incompatible index.pkl. Please re-run 'create_faiss_index.py'.")
        return None, None, None

    # 3. Initialize LLM (LlamaCpp)
    print(f"Loading LLM from: {GGUF_MODEL_PATH}")
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"Error: GGUF model not found at '{GGUF_MODEL_PATH}'.")
        print("Please ensure your Mistral GGUF model is in the specified path.")
        return None, None, None
    if not os.path.exists(LLAMA_CPP_EXECUTABLE_PATH):
        print(f"Error: llama.cpp executable not found at '{LLAMA_CPP_EXECUTABLE_PATH}'.")
        print("Please ensure llama.cpp is built and 'main' executable is in the specified path.")
        return None, None, None

    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=512, # Batch size for prompt processing
            n_ctx=2048,  # Context window size
            f16_kv=True, # Use FP16 for key/value cache
            verbose=False, # Set to True for more verbose LLM output
            # Add other llama.cpp parameters as needed
            # e.g., temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1
        )
        print("Mistral GGUF LLM loaded.")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Please check your GGUF model path, llama.cpp executable, and n_gpu_layers setting.")
        return None, None, None

    return vectorstore, embeddings, llm # Return embeddings as well, needed for similarity_search_with_score

def process_query(query, vectorstore, embeddings, llm):
    """Processes a user query using RAG with a fallback to LLM-only,
       and filters retrieved documents by FAISS similarity score."""
    print(f"\nProcessing query: '{query}'")

    # Perform similarity search with scores
    # Note: FAISS.similarity_search_with_score returns a list of (Document, score) tuples
    # Lower score (L2 distance) means higher similarity.
    print(f"Retrieving top {INITIAL_RETRIEVAL_COUNT} documents with scores...")
    retrieved_with_scores = vectorstore.similarity_search_with_score(query, k=INITIAL_RETRIEVAL_COUNT)

    # Filter documents based on SCORE_THRESHOLD
    filtered_docs = []
    for doc, score in retrieved_with_scores:
        if score <= SCORE_THRESHOLD:
            filtered_docs.append(doc)
            print(f"  - Document added (Score: {score:.4f}): {doc.page_content[:50]}...")
        else:
            print(f"  - Document skipped (Score: {score:.4f} > Threshold {SCORE_THRESHOLD:.4f}): {doc.page_content[:50]}...")

    if len(filtered_docs) >= FALLBACK_DOC_THRESHOLD:
        print(f"Found {len(filtered_docs)} relevant documents (after score filtering). Using RAG chain.")

        # Create a retriever from the filtered documents for the RAG chain
        # We need to create a temporary FAISS index from these filtered docs to use .as_retriever()
        # A simpler way is to manually pass context, but using RetrievalQA needs a retriever.
        # For simplicity and to leverage existing chain, we'll create a temporary in-memory FAISS from filtered docs
        # Or, more directly, we can build the chain_type_kwargs context from filtered_docs.

        # Option 1: Create a temporary in-memory FAISS from filtered_docs (more robust for complex chains)
        # This requires re-embedding the filtered docs, which is inefficient if many.
        # Temp_vectorstore = FAISS.from_documents(filtered_docs, embeddings)
        # retriever = Temp_vectorstore.as_retriever(search_kwargs={"k": len(filtered_docs)}) # Use all filtered docs

        # Option 2: Manually construct the context for the prompt (simpler for 'stuff' chain)
        context_text = "\n\n".join([doc.page_content for doc in filtered_docs])

        # Create RAG chain using prompt template and context
        # We're effectively doing the "stuff" chain manually here for precise control
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        final_prompt = rag_prompt.format(context=context_text, question=query)

        answer = llm.invoke(final_prompt) # Directly invoke LLM with formatted RAG prompt

        print("\n--- Answer (RAG) ---")
        print(answer)
        print("\n--- Source Documents (Filtered by Score) ---")
        if filtered_docs:
            for i, doc in enumerate(filtered_docs):
                print(f"Document {i+1} (Score: {retrieved_with_scores[retrieved_with_scores.index((doc, [s for d,s in retrieved_with_scores if d==doc][0]))][1]:.4f}):") # Re-find score for printing
                print(f"  Source: {doc.metadata.get('source', 'N/A')}")
                print(f"  Content (excerpt): {doc.page_content[:200]}...") # Print first 200 chars
        else:
            print("No specific source documents were associated with this answer after filtering.")

    else:
        print(f"Found {len(filtered_docs)} relevant documents (after score filtering, less than threshold {FALLBACK_DOC_THRESHOLD}). Falling back to LLM-only.")
        # Fallback to LLM-only
        llm_only_prompt = PromptTemplate.from_template(LLM_ONLY_PROMPT_TEMPLATE)
        formatted_prompt = llm_only_prompt.format(question=query)
        answer = llm.invoke(formatted_prompt)

        print("\n--- Answer (LLM Only) ---")
        print(answer)
        print("\n--- Note ---")
        print("No sufficiently relevant documents were found in the knowledge base for this query. The answer is based solely on the LLM's pre-trained knowledge.")

def main():
    vectorstore, embeddings, llm = initialize_rag_system()

    if vectorstore is None or llm is None or embeddings is None: # Check for embeddings too
        print("Failed to initialize RAG system. Exiting.")
        return

    print("\n--- RAG System Ready ---")
    print("Enter your query (or 'quit' to exit):")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'quit':
            print("Exiting RAG system. Goodbye!")
            break
        if not query:
            print("Please enter a query.")
            continue

        try:
            process_query(query, vectorstore, embeddings, llm) # Pass embeddings to process_query

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            print("Please check your LLM and FAISS setup.")

if __name__ == "__main__":
    main()
