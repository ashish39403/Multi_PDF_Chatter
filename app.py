import streamlit as st
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder , PromptTemplate
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

def get_text_input(pdf_docs):
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text= ""
        for pages in pdf_reader.pages:
            text+= pages.extract_text()
    return text
            
def get_text_split(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, 
        chunk_overlap= 100
    )
    chunks= text_splitter.split_text(text)
    return chunks      


 
def get_vector_store(chunks):
    embeddings=OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vectorstore = FAISS.from_texts(chunks , embedding=embeddings)
    return vectorstore




def get_conversational_chain(vector_store):
  
    
    llm = ChatOllama(model="llama3:8b" , temperature=0.4)
    memory =ConversationBufferMemory(memory_key="chat_history" , return_messages=True)
    conversation_chain =ConversationalRetrievalChain.from_llm(
        
        llm= llm, 
        retriever =vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.session_state.chat_history= response['chat_history']
    
    for i ,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(message.content , f"{i}👦user")
        else:
            
            st.write(message.content , f"{i}🤖ai")

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history"not in st.session_state:
        st.session_state.chat_history= None
        
        
    st.title('Chat With Mutlitple PDF..')
    user_question=st.text_input('Ask Your Query Here...')
    if user_question:
        handle_user_input(user_question)
   
        
     
        
    with st.sidebar:
        user_file = st.file_uploader('Upload Your File Here...' , accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing..'):
                
                #getting the text
                raw_text = get_text_input(user_file)
                #get the text in chunks
                text_chunks = get_text_split(raw_text)
                #get the vector store
                vector_store = get_vector_store(text_chunks)
                #get converation chain
                st.session_state.conversation = get_conversational_chain(vector_store)
                


if __name__=="__main__":
    main()