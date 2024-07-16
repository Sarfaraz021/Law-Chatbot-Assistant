import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader, PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from prompt import prompt_template_text, toc_prompt_template_text
from langchain_google_community import GoogleDriveLoader
from fpdf import FPDF
from langchain_community.document_loaders import UnstructuredFileIOLoader


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None
        self.folder_id = "1LiU9GgzXHamR8MaDk5a3ygFFIYWLP3tX"
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

        self.cred_relative_path = '.credentials'
        self.cred_filename = 'credentials.json'
        self.cred_absolute_path = os.path.join(
            self.cred_relative_path, self.cred_filename)

        self.token_relative_path = '.credentials'
        self.token_filename = 'google_token.json'
        self.token_absolute_path = os.path.join(
            self.token_relative_path, self.token_filename)
        self.initialize_gdrive_retriever(self.folder_id)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_prompt_template(self):
        """Sets up the prompt template for chat completions."""
        self.template = prompt_template_text
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=self.template,
        )

    def initialize_gdrive_retriever(self, folder_id):
        """Initializes the retriever with documents from Google Drive."""
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            token_path=self.token_absolute_path,
            credentials_path=self.cred_absolute_path,
            recursive=False,
            file_loader_cls=UnstructuredFileIOLoader,
            file_loader_kwargs={"mode": "elements"},
            template="gdrive-all-in -folder",
            num_results=10,


        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        print(docs)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1-aws')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def finetune_gdrive(self, folder_id):
        """Fine-tunes the assistant with documents from a Google Drive folder."""
        self.initialize_gdrive_retriever(folder_id)
        return "Fine-tuning with Google Drive folder done successfully. You can now chat with your updated context from Google Drive."

    def chat(self, user_input):
        """Starts a chat session with the AI assistant."""
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={"verbose": False, "prompt": self.prompt_template,
                               "memory": ConversationBufferMemory(context="context", memory_key="history", input_key="question")}
        )

        assistant_response = chain.invoke(user_input)  # type: ignore
        response_text = assistant_response['result']
        return response_text


def main():
    assistant = RAGAssistant()

    st.set_page_config(page_title="Gerry-Law-Chatbot-Assistant", layout="wide")
    st.title("Gerry Law Chat Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gdrive_folder_id" not in st.session_state:
        st.session_state.gdrive_folder_id = ""
    if "toc_content" not in st.session_state:
        st.session_state.toc_content = ""

    option = st.sidebar.selectbox(
        "Choose an option", ("Chat", "Fine-tuning", "Fine-tune with Google Drive", "Generate Table of Contents"))

    if option == "Chat":
        st.header("Chat with your Docs")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter your message:"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = assistant.chat(prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

    elif option == "Fine-tune with Google Drive":
        st.header("Fine-tune with Google Drive")
        folder_id = st.text_input("Enter Google Drive folder ID")

        if st.button("Fine-tune"):
            if folder_id:
                with st.spinner("Fine-tuning with Google Drive in progress..."):
                    message = assistant.finetune_gdrive(folder_id)
                st.success(message)


if __name__ == "__main__":
    main()
