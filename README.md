
# Domain-Specific RAG Chatbot with LLM Fallback (LLM + FAISS)

This project is a **Retrieval-Augmented Generation (RAG)** system using a fine-tuned large language model (LLM) combined with a custom knowledge base and FAISS indexing for fast vector retrieval. It is designed to serve as a domain-specific assistant for internal company use, and is currently being implemented for real user support.

## Note
This jupyter notebook only contains a generate_answer() method which is NOT a chatbot experience, using llama.cpp would be the preferred next step to create a CLI venv chatot you can use, HuggingFace also provides 'Spaces' where you can host using gradio or streamlit

## Features

- **Fine-tuned LLM** for custom domain language and task adaptation  
- **FAISS vector store** for fast semantic search over documents  
- **Retrieval-Augmented Generation** for grounded, context-aware responses  
- **Modular design** — easily adaptable for other domains or use cases

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/kpis-him/LRag
cd LRag
````

### Run the Notebook

Open the notebook:

```bash
jupyter notebook notebooks/RAGmodel.ipynb
```

Then follow the cells to:

* Load the FAISS index and fine-tuned model
* Input a natural language query
* Retrieve relevant context from the knowledge base
* Generate an informed response

---

## Example Query

**Input:**

> "How do I configure the device for remote diagnostics?"

**Generated Response:**

> "To configure remote diagnostics, ensure the firmware is up-to-date and enable the telemetry service in the admin console. Refer to section 4.2 of the setup guide for more details."

---

## Real-World Use

This system is currently being used in an internal capacity by engineers for responding to client questions in a telecom domain. The core RAG engine is under consideration for wider deployment within industry workflows.

---

## Future Work

* Wrap into a Streamlit app for interactive querying
* Add document upload and indexing UI
* Explore multilingual retrieval and response generation
* Expand to other technical domains (e.g., biomedical, manufacturing)

---

## Credits

Created by Kushal Patil as part of an independent AI engineering project.
Deployed and iteratively improved with real-world use case feedback.

---

## 📜 License

Open-source under the MIT License. See [LICENSE](LICENSE) for details.

