
# --- Step 1: Import necessary libraries ---
import re
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
# import google.generativeai as genai # Uncomment when using the actual API

# --- Step 2: Define all functions ---

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a specified PDF file.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The extracted text from the PDF.
    """
    print(f"Extracting text from: {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        print("Text extraction successful.")
        return text
    except FileNotFoundError:
        print(f"Error: The file at {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during PDF extraction: {e}")
        return None

def clean_text(text):
    """
    Removes noise, unwanted characters, and extra whitespace from the text.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    print("Cleaning text...")
    # Remove headers, footers, and page numbers (customize regex as needed)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # A simple way to remove initial lines that might be headers
    text = re.sub(r'^(.*?)\n', '', text, count=5)

    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    print("Text cleaning complete.")
    return text

def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """
    Splits text into smaller, overlapping chunks.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The desired size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    print("Chunking text...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_database(chunks, model):
    """
    Creates a FAISS vector database from a list of text chunks.

    Args:
        chunks (list): The list of text chunks.
        model: The sentence transformer model for encoding.

    Returns:
        faiss.Index: The FAISS index containing the vector embeddings.
    """
    print("Creating vector embeddings...")
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)
    
    print("Building FAISS index...")
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings).astype('float32'))
    print("Vector database created successfully.")
    return index

def search_relevant_chunks(query, index, chunks, model, top_k=5):
    """
    Finds the most relevant text chunks for a given query from the FAISS index.

    Args:
        query (str): The user's question.
        index (faiss.Index): The FAISS vector database.
        chunks (list): The original list of text chunks.
        model: The sentence transformer model.
        top_k (int): The number of top relevant chunks to retrieve.

    Returns:
        list: A list of the most relevant text chunks.
    """
    print(f"Searching for top {top_k} relevant chunks...")
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    print("Search complete.")
    return relevant_chunks

def generate_answer(query, context):
    """
    Generates a helpful answer using the Gemini model based on the retrieved context.
    NOTE: This is a placeholder and does not make a real API call.

    Args:
        query (str): The user's question.
        context (list): A list of relevant text chunks.

    Returns:
        str: The generated answer.
    """
    print("Generating final answer...")
    prompt = f"""
    You are a helpful legal assistant. Based on the following context from a legal document, 
    please answer the user's question. If the context does not contain the answer,
    state that you cannot answer based on the provided information.

    Context:
    {" ".join(context)}

    Question:
    {query}

    Answer:
    """
    
    # --- UNCOMMENT THE FOLLOWING LINES TO USE THE ACTUAL GEMINI API ---
    # try:
    #     genai.configure(api_key="YOUR_API_KEY")
    #     model = genai.GenerativeModel('gemini-pro')
    #     response = model.generate_content(prompt)
    #     return response.text
    # except Exception as e:
    #     return f"An error occurred while calling the Gemini API: {e}"
    # ----------------------------------------------------------------

    # Placeholder for the response, as we can't make a real API call here.
    return "Based on the provided legal context, you should report the theft to the police. They will file a First Information Report (FIR) under the relevant sections of the Indian Penal Code, such as Section 379 for theft."

# --- Step 3: Main execution block ---

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace this with the actual path to your PDF file.
    PDF_FILE_PATH = 'C:\Users\adity\OneDrive\Desktop\New folder (3)\Pradeep_Tomar_And_Another_vs_State_Of_U_P_And_Another_on_27_January_2021.PDF' 
    USER_QUERY = "My phone was stolen, what can I do?"

    print("--- Starting RAG Model Pipeline ---")

    # 1. Extract Text
    # Create a dummy PDF file for demonstration if it doesn't exist
    try:
        with open(PDF_FILE_PATH, 'rb') as f:
            pass
    except FileNotFoundError:
        print(f"'{PDF_FILE_PATH}' not found. A dummy file will not be created. Please provide a valid PDF file.")
        # In a real scenario, you would handle this error more gracefully.
        # For this script, we'll exit if the file doesn't exist.
        exit()

    raw_text = extract_text_from_pdf(PDF_FILE_PATH)

    if raw_text:
        # 2. Clean Text
        cleaned_text = clean_text(raw_text)

        # 3. Chunk Text
        text_chunks = chunk_text(cleaned_text)

        # 4. Load Embedding Model
        print("Loading sentence transformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

        # 5. Create Vector Database
        vector_db = create_vector_database(text_chunks, embedding_model)

        # 6. Retrieve Relevant Chunks
        relevant_context = search_relevant_chunks(USER_QUERY, vector_db, text_chunks, embedding_model)

        print("\n--- Retrieved Context ---")
        for i, chunk in enumerate(relevant_context):
            print(f"Chunk {i+1}:\n{chunk}\n")

        # 7. Generate Final Answer
        final_answer = generate_answer(USER_QUERY, relevant_context)

        print("\n--- Final Generated Answer ---")
        print(final_answer)
    
    print("\n--- RAG Model Pipeline Finished ---")
