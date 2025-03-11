import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Initialize the vector store and similarity threshold
if "vector_store" not in st.session_state:
    st.session_state.vector_store = {}
similarity_threshold = 0.7

# Initialize the embedding model
@st.cache_data(ttl=3600)
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

def get_embedding(query):
    """Generate embedding for a query."""
    model = load_model()
    return model.encode([query])[0]

def search_similar_vectors(query_embedding):
    """Search for similar vectors in the vector store."""
    similar_vectors = []
    for stored_query, stored_embedding in st.session_state.vector_store.items():
        if stored_query.endswith("_response"):
            continue
        
        similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        if similarity > similarity_threshold:
            similar_vectors.append((stored_query, similarity))
    return similar_vectors

def add_to_vector_store(query, response):
    """Add a query and its response to the vector store."""
    query_embedding = get_embedding(query)
    st.session_state.vector_store[query] = query_embedding
    # Store the response (for simplicity, just store it as a string)
    st.session_state.vector_store[f"{query}_response"] = response

def get_response_from_cache(query):
    """Get a response from the cache if a similar query exists."""
    query_embedding = get_embedding(query)
    if query in st.session_state.vector_store:
        # If the exact query is in the cache, return its response
        return st.session_state.vector_store[f"{query}_response"]
    
    similar_vectors = search_similar_vectors(query_embedding)
    if similar_vectors:
        # Use the response associated with the top similar vector
        top_query = max(similar_vectors, key=lambda x: x[1])[0]
        return st.session_state.vector_store[f"{top_query}_response"]
    else:
        return None

def call_llm(query):
    """Call OpenAI API to get a response."""
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    try:
        # Use the Completion endpoint for text generation
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=query,
            max_tokens=50,
            temperature=0.0
        )
        return response.choices[0].text
    except Exception as e:
        return f"Error calling LLM: {e}"

def main():
    st.title("Semantic Caching Demo")
    query = st.text_input("Enter your query")
    
    if st.button("Submit"):
        cached_response = get_response_from_cache(query)
        if cached_response and not cached_response.startswith("Error calling LLM:"):
            st.write(f"Response from cache: {cached_response}")
        else:
            try:
                llm_response = call_llm(query)
                add_to_vector_store(query, llm_response)
                st.write(f"Response from LLM: {llm_response}")
            except Exception as e:
                st.error(f"Error calling LLM: {e}")

if __name__ == "__main__":
    main()
