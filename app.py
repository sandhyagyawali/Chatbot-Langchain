import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
  
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=80,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
  load_dotenv()
  st.set_page_config(page_title="Chat with multiple PDFS",page_icon="books:")
  st.header("Chat with multiple PDFS:books:")
  st.text_input("Asks a question about your documents")
  
  
  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your pdfs here and click on `Process`",accept_multiple_files=True)
    if st.button("Process"):
      with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # # create conversation chain
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)

if __name__ == '__main__':
    main()