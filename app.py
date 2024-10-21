import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str, previous_context: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Combine previous context with the new context
    full_context = f"{previous_context}\n\n{context_text}" if previous_context else context_text
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=full_context, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, full_context

def main():
    st.title("RAG-based Chat Interface")
    previous_context = ""
    
    # Input for user query
    query_text = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        if query_text:
            response, previous_context = query_rag(query_text, previous_context)
            st.write(response)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
