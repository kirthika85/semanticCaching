import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from openai import OpenAI

# Initialize the vector store and similarity threshold
if "vector_store" not in st.session_state:
    st.session_state.vector_store = {}
similarity_threshold = 0.7

api_key=st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Initialize the embedding model
@st.cache_data(ttl=3600)
def load_model():
    print("Loading SentenceTransformer model...")
    return SentenceTransformer("all-mpnet-base-v2")

def get_embedding(query):
    """Generate embedding for a query."""
    print(f"Generating embedding for query: {query}")
    model = load_model()
    embedding = model.encode([query])[0]
    print(f"Generated embedding: {embedding}")
    return embedding

def search_similar_vectors(query_embedding):
    """Search for similar vectors in the vector store."""
    print("Searching for similar vectors in the vector store...")
    similar_vectors = []
    for stored_query, stored_embedding in st.session_state.vector_store.items():
        if stored_query.endswith("_response"):
            continue
        
        similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        print(f"Similarity for query '{stored_query}': {similarity}")
        if similarity > similarity_threshold:
            similar_vectors.append((stored_query, similarity))
            print(f"Added '{stored_query}' to similar vectors with similarity {similarity}")
    print(f"Found {len(similar_vectors)} similar vectors")
    return similar_vectors

def add_to_vector_store(query, response):
    """Add a query and its response to the vector store."""
    print(f"Adding query '{query}' and response '{response}' to the vector store...")
    query_embedding = get_embedding(query)
    st.session_state.vector_store[query] = query_embedding
    st.session_state.vector_store[f"{query}_response"] = response
    print(f"Added query '{query}' with embedding {query_embedding} and response '{response}'")

def get_response_from_cache(query):
    """Get a response from the cache if a similar query exists."""
    print(f"Checking cache for query '{query}'...")
    query_embedding = get_embedding(query)
    if query in st.session_state.vector_store:
        print(f"Exact query '{query}' found in cache")
        return st.session_state.vector_store[f"{query}_response"]
    
    similar_vectors = search_similar_vectors(query_embedding)
    if similar_vectors:
        print(f"Found similar vectors for query '{query}'")
        top_query = max(similar_vectors, key=lambda x: x[1])[0]
        print(f"Using response from top similar query '{top_query}'")
        return st.session_state.vector_store[f"{top_query}_response"]
    else:
        print(f"No similar queries found for '{query}'")
        return None

def call_llm(query):
    """Call OpenAI API to get a response."""
    print(f"Calling LLM for query '{query}'...")
    try:
        # Use the Completion endpoint for text generation with a conversational prompt
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in answering user questions"},
                {"role": "user", "content": query}
            ],
            temperature=0.0
       )
        print(f"Received response from LLM: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error calling LLM: {e}"

def main():
   
    st.title("Semantic Caching Demo")
    query = st.text_input("Enter your query")
    
    if st.button("Submit"):
        cached_response = get_response_from_cache(query)
        if cached_response and not cached_response.startswith("Error calling LLM:"):
            print(f"Using cached response for query '{query}'")
            st.write(f"Response from cache: {cached_response}")
        else:
            try:
                llm_response = call_llm(query)
                add_to_vector_store(query, llm_response)
                print(f"Added query '{query}' and response '{llm_response}' to cache")
                st.write(f"Response from LLM: {llm_response}")
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                st.error(f"Error calling LLM: {e}")

if __name__ == "__main__":
    main()
