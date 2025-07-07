# Intelligent Complaint Analysis for Financial Services

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a Retrieval-Augmented Generation (RAG) solution to analyze customer complaints for **CrediTrust Financial**. The system uses a chatbot interface to transform unstructured complaint data into actionable insights, enabling teams to quickly understand customer pain points.

The analysis focuses on five key product areas:
*   Credit Cards
*   Personal Loans
*   Buy Now, Pay Later (BNPL)
*   Savings Accounts
*   Money Transfers

## ✨ Key Features

-   **Automated Data Cleaning**: Preprocesses raw complaint data to improve quality for analysis.
-   **Semantic Search**: Converts complaints into vector embeddings for powerful, meaning-based search.
-   **RAG-Powered Q&A**: Allows users to ask natural language questions about the complaints and receive context-aware answers.
-   **Persistent Vector Store**: Uses ChromaDB to store and manage embeddings efficiently.

## ⚙️ System Workflow

The project follows a standard RAG pipeline:

1.  **Ingestion & Preprocessing**: Raw complaint data is loaded, cleaned, and filtered.
2.  **Chunking & Embedding**: Cleaned text is broken into smaller, overlapping chunks, which are then converted into numerical vectors (embeddings) using a sentence-transformer model.
3.  **Indexing**: The embeddings and their corresponding text are stored in a ChromaDB vector store.
4.  **Retrieval & Generation**:
    -   A user asks a question (e.g., "What are the main issues with money transfers?").
    -   The system retrieves the most relevant chunks from the vector store.
    -   The question and the retrieved context are passed to a Large Language Model (LLM) to generate a human-like answer.

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/intelligent-complaint-analysis.git
    cd intelligent-complaint-analysis
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up the data:**
    -   Obtain the CFPB complaints dataset.
    -   Place it in the `data/` directory with the filename `complaints.csv`. (Note: This directory is ignored by Git).

### How to Use

1.  **Process the Data and Build the Vector Store:**
    Run the main processing script. This will perform the EDA, cleaning, chunking, embedding, and indexing steps.
    ```sh
    python src/build_vector_store.py
    ```
    This script will generate `data/filtered_complaints.csv` and populate the `vector_store/` directory.

2.  **Run the Question-Answering Chatbot:**
    Launch the interactive chatbot interface (assuming a Streamlit or similar app).
    ```sh
    streamlit run app.py
    ```
    You can now ask questions about the complaint data in your browser.

## 📁 Project Structure

```
.
├── data/               # Raw and processed datasets (gitignored)
├── notebooks/          # Jupyter notebooks for EDA and prototyping
├── reports/            # Project reports and documentation
├── src/                # Source code for the RAG pipeline
│   ├── preprocessing.py
│   ├── chunk_and_embed.py
│   └── build_vector_store.py
├── tests/              # Unit and integration tests
├── vector_store/       # Persisted ChromaDB vector database
├── app.py              # Streamlit chatbot application
├── requirements.txt    # Project dependencies
└── README.md
```

## 🛠️ Implementation Details

### Task 1: EDA and Preprocessing

The goal was to prepare the raw complaint data for the RAG pipeline.

-   **Analysis**: The initial dataset was analyzed to understand complaint distribution by product and narrative length.
-   **Filtering**: The data was filtered to retain only the five target product categories and to remove records with empty complaint narratives.
-   **Text Cleaning**: Narratives were standardized by lowercasing, removing boilerplate text ("In accordance with federal law..."), and stripping special characters and excess whitespace.
-   **Output**: The cleaned dataset is saved to `data/filtered_complaints.csv`, and the full process is documented in `notebooks/eda_preprocessing.ipynb`.

### Task 2: Chunking, Embedding, and Indexing

This task converts the cleaned text into a searchable vector index.

-   **Chunking Strategy**: We used `RecursiveCharacterTextSplitter` from LangChain with a `chunk_size` of 300 and `chunk_overlap` of 50. This strategy balances retaining semantic context within a chunk while ensuring the chunks are small enough for effective retrieval.

-   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` was chosen for its excellent balance of performance, speed, and size. It is a lightweight but powerful open-source model optimized for semantic similarity tasks.

-   **Indexing Process**: Each text chunk was embedded and stored in a **ChromaDB** vector store. Metadata, including the original complaint ID and product category, was stored alongside each vector to allow for easy traceability and filtered searches. The database is persisted locally in the `vector_store/` directory.

## 📈 Future Work

-   **Advanced RAG**: Implement more sophisticated retrieval techniques, such as re-ranking or query expansion.
-   **Evaluation Framework**: Build a robust evaluation pipeline to measure the accuracy and relevance of the generated answers.
-   **UI Enhancements**: Add features to the user interface for filtering by product or date range.
-   **Model Fine-Tuning**: Experiment with fine-tuning an embedding model on the specific financial complaint domain.