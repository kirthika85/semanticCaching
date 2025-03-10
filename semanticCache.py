import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the vector store and similarity threshold
vector_store = {}
similarity_threshold = 0.7

# Initialize the embedding model
#model = SentenceTransformer("all-mpnet-base-v2")
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

def get_embedding(query):
    """Generate embedding for a query."""
    return model.encode([query])[0]

def search_similar_vectors(query_embedding):
    """Search for similar vectors in the vector store."""
    similar_vectors = []
    for stored_query, stored_embedding in vector_store.items():
        similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        if similarity > similarity_threshold:
            similar_vectors.append((stored_query, similarity))
    return similar_vectors

def add_to_vector_store(query, response):
    """Add a query and its response to the vector store."""
    query_embedding = get_embedding(query)
    vector_store[query] = query_embedding
    # Store the response (for simplicity, just store it as a string)
    vector_store[f"{query}_response"] = response

def get_response_from_cache(query):
    """Get a response from the cache if a similar query exists."""
    query_embedding = get_embedding(query)
    similar_vectors = search_similar_vectors(query_embedding)
    if similar_vectors:
        # Use the response associated with the top similar vector
        top_query = max(similar_vectors, key=lambda x: x[1])[0]
        return vector_store[f"{top_query}_response"]
    else:
        return None

def call_llm(query):
    """Simulate calling an LLM (replace with actual LLM call)."""
    # For demonstration purposes, just return a generic response
    return f"LLM response for: {query}"

def main():
    st.title("Semantic Caching Demo")
    query = st.text_input("Enter your query")
    
    if st.button("Submit"):
        cached_response = get_response_from_cache(query)
        if cached_response:
            st.write(f"Response from cache: {cached_response}")
        else:
            llm_response = call_llm(query)
            add_to_vector_store(query, llm_response)
            st.write(f"Response from LLM: {llm_response}")

if __name__ == "__main__":
    main()
