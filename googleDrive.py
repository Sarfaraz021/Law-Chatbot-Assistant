from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# from langchain_community.document_loaders import GoogleDriveLoader
from langchain_google_community import GoogleDriveLoader
import os
from dotenv import load_dotenv


# Ensure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# credentials_path = "D:\\Gerry-Law-Chatbot-Assistant\\.credentials\\credentials.json"
cred_relative_path = '.credentials'
cred_filename = 'credentials.json'
credentials_path = os.path.join(cred_relative_path, cred_filename)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path


os.environ["OPENAI_API_KEY"] = "sk-proj-lPoqX8DjLpfp56GjykkMT3BlbkFJDT8dYbKFqUuZpYp9DAhe"
folder_id = "1-2-LSNoT5UVgU83BD9Gy1LAXyqVFAceO"


class Main:
    def __init__(self):
        self.load_env_variables()
        self.retriever = None
        self.initialize_retriever(folder_id)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def initialize_retriever(self, folder_id):
        """Initializes the retriever with documents from the specified directory path."""
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            recursive=False
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def chat_drive(self):
        if self.retriever is None:
            print("Retriever is not initialized.")
            return

        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=self.retriever)

        while True:
            query = input("> ")
            answer = chain.invoke(query)
            print(f"AI Assistant: {answer['result']}")
            print("*********************************")


if __name__ == "__main__":
    m = Main()
    m.chat_drive()
