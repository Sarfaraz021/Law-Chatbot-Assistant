from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import GoogleDriveLoader
import os


os.environ["OPENAI_API_KEY"] = "sk-proj-lPoqX8DjLpfp56GjykkMT3BlbkFJDT8dYbKFqUuZpYp9DAhe"
folder_id = "1B9CPHDWp3yZa6URSSMQIThrpr52gMtiF"
loader = GoogleDriveLoader(
    folder_id=folder_id,
    recursive=False
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=200, separators=[" ", ",", "\n"]
)

texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever)

while True:
    query = input("> ")
    answer = qa.run(query)
    print(answer)
