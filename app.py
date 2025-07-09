"""
app.py

Task 4: Interactive Chat Interface for RAG System using Gradio
- Allows users to ask questions about customer complaints
- Displays AI-generated answer and the retrieved sources
- Includes a clear button and supports chatbot-style conversation
- Streams the answer for better UX (if supported by the LLM)
"""
import sys
import os
import gradio as gr
from transformers import pipeline

# Ensure local src/ is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from rag_pipeline import rag_answer

# Initialize the text generation model
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def chat_fn(message, history):
    """
    Given a user message and chat history, runs the RAG pipeline to get an answer and sources.
    Returns a formatted string with the answer and sources.
    """
    result = rag_answer(message)
    answer = result["answer"]
    sources = result["retrieved_sources"]
    sources_str = "\n".join([f"- [{meta.get('product', '')}] {doc}" for doc, meta in sources])
    full_response = answer + "\n\n**Sources:**\n" + sources_str
    return full_response

def respond(message, chat_history):
    """
    Handles a new user message: gets the RAG answer, appends to chat history, and returns updated state.
    Returns chat history in Gradio 'messages' format.
    """
    response = chat_fn(message, chat_history)
    # Gradio 'messages' format: list of dicts with 'role' and 'content'
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    return "", chat_history

def clear_chat():
    """
    Clears the chat input and history.
    """
    return "", []

with gr.Blocks() as demo:
    gr.Markdown("""
    # CrediTrust Complaint Insights Chatbot
    Ask any question about customer complaints. The AI will answer and show the most relevant complaint excerpts as sources.
    """)
    # Use type='messages' for OpenAI-style chat format (future-proof)
    chatbot = gr.Chatbot(type='messages')
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your question and press Enter")
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    txt.submit(respond, [txt, chatbot], [txt, chatbot])
    submit_btn.click(respond, [txt, chatbot], [txt, chatbot])
    clear_btn.click(clear_chat, None, [txt, chatbot], queue=False)

if __name__ == "__main__":
    demo.launch(share=True)