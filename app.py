import streamlit as st
import os
from  PyPDF2 import PdfReader
import docx 
from  langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
import random 
from datetime import datetime
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import string 
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

openai_key = st.secrets =["OPEN API KEY "]
qdrant_url = st.secrets =["OPEN API KEY "]
qdrant_api_key = st.secrets =["OPEN API KEY"]
 
def main():
    load_dotenv()
    st.set_page_config(page_title= "Q/A with your file ") 
    st.header("Retrieval QA Chain")

    if "Conversation" not in st.session_state:
        st.session_state.Conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

     
         

    if "processcomplete " not in st.session_state:
        st.session_state.processcomplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("upload your file", type=['pdf'], accept_multiple_files=True)
        openai_api_key = openai_key

        process = st.button=("Process")  

    if process:
        if not openai_api_key:
            st.info("Please add you openAI API key to continouos ")          
            st.stop()

        text_chuck_list =[]
        for uploaded_file in uploaded_files:
            file_name =uploaded_file.name 
            file_text = get_files_text(uploaded_file) 

            text_chucks =get_text_chucks(file_text, file_name)
            text_chuck_list.extend(text_chucks)  

            #create vetor store
        curr_date =str(datetime.now())
        collection_name =" ".join(random.choices(string.ascii_letters, k=4)) + curr_date.split('.')[0].replace(':','-').replace(" ", " T")
        vetorestore = get_vectorstore(text_chuck_list, collection_name)
        st.write("Vector Store Create ....")
        num_chuck =4

        st.session_state.Conversation = get_qa_chain(vetorestore, num_chuck)

        st.session_state.processcomplete =True


    if  st.session_state.processcomplete ==True:
        user_question = st.chat_input("ASK question about your files.") 
        if user_question:
            handel_userinput(user_question)



def  get_files_text(uploaded_file):
    text =" "
    split_tup = os.path.splitext(uploaded_file.name)
    file_extensive = split_tup[1]

    if file_extensive =='pdf':
        text += get_pdf_text(uploaded_file)
    elif file_extensive == 'docx':
        text += get_docx_text(uploaded_file)
    else:
        pass

    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text =" "

    for page in pdf_reader.pages:
        text += page.extract_text()
    return text 

def get_docx_text (file):
    doc =docx.Document=file
    allText =[]
    for docpara in doc.paragraphys:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text 

def get_text_chucks (text, filename):

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chuck_size= 80,
        chuck_overlap =20,
        length_function = len
    )
    chucks = text_splitter.split_text(text)
    doc_list= []
    for chuck in chucks:
        metadata ={ "source": filename}
        doc_string = Document(page_content=chuck , metadata=metadata)
        doc_list.append(doc_string)

    return doc_list


def  get_vectorstore(text_chucks, COLLECTION_NAME):

    try:

        knowledge_base = Qdrant.from_documents(
            documents = text_chucks,
            embedding = embeddings,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
        )

      

    except Exception as e:
        st.write(f"Error :{e}")    

    return knowledge_base


def get_qa_chain (vetctor_store, num_chucks):

    
                               
      qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-3.5-turbo"), chain_type="stuff",
                                retriever=vetctor_store.as_retriever(search_type="similarity",
                                                            search_kwargs={"k": num_chucks}),  return_source_documents=True)
      return qa 


def  handel_userinput (user_question):
    with st.spinner('Generating response....'):
        result= st.session_state.conversation({"query": user_question})
        response = result ['result']
        source = result ['source_documents'][0].metadata['source']

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(f"{response} \n Source Document: {source}")


#layout of input response 
response_container = st.container()

with response_container:
    for i, message in enumerate(st.session_state.chat_history ):
         if i % 2 == 0:
                message(message, is_user=True, key=str(i))
         else:
                message(message, key=str(i))


if __name__ == '__main__':
    main()
        
   
                                         


