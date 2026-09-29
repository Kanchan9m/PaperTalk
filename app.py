import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaLLM

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")

    # print(os.getenv("OPENAI_API_KEY"))

    st.header("Ask your PDF")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if(pdf != None):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
           text += page.extract_text()

        # st.write(text)
        text_splitter = CharacterTextSplitter(
            separator="\n", 
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        document = FAISS.from_texts(chunks, embeddings)
        document.save_local("faiss_index")
        knowledge_base = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OllamaLLM(model="llama3.2:3b")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
            

if __name__ == "__main__":
    main()