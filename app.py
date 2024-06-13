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
import io


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None
        self.relative_path = 'data'
        self.filename = 'dummy.txt'
        self.absolute_path = os.path.join(self.relative_path, self.filename)
        self.initialize_retriever(self.absolute_path)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def setup_prompt_template(self):
        """Sets up the prompt template for chat completions."""
        self.template = prompt_template_text
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=self.template,
        )
        self.toc_template = toc_prompt_template_text
        self.toc_prompt_template = PromptTemplate(
            input_variables=["question"],
            template=self.toc_template,
        )

    def initialize_retriever(self, directory_path):
        """Initializes the retriever with documents from the specified directory path."""
        loader = TextLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def finetune(self, file_path):
        """Determines the document type and uses the appropriate loader to fine-tune the model."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load_and_split() if hasattr(
            loader, 'load_and_split') else loader.load()

        self.process_documents(documents)

    def process_documents(self, documents):
        """Process and index the documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def chat(self, user_input):
        """Starts a chat session with the AI assistant."""
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={"verbose": False, "prompt": self.prompt_template,
                               "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
        )

        assistant_response = chain.invoke(user_input)  # type: ignore
        response_text = assistant_response['result']
        return response_text

    def initialize_gdrive_retriever(self, folder_id):
        """Initializes the retriever with documents from Google Drive."""
        loader = GoogleDriveLoader(folder_id=folder_id, recursive=False)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def finetune_gdrive(self, folder_id):
        """Fine-tunes the assistant with documents from a Google Drive folder."""
        self.initialize_gdrive_retriever(folder_id)
        return "Fine-tuning with Google Drive folder done successfully. You can now chat with your updated context from Google Drive."

    def generate_toc(self, user_input):
        """Generates a table of contents based on user input."""
        toc_chain = LLMChain(
            llm=self.llm,
            prompt=self.toc_prompt_template
        )
        toc_response = toc_chain.invoke({"question": user_input})
        return toc_response['text']

    def save_toc_as_pdf(self, toc_content):
        """Saves the table of contents as a PDF file and returns it as bytes."""
        font_path = os.path.join(os.path.dirname(
            __file__), 'fonts')  # Specify the font directory
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Register the Arial Unicode MS font
        pdf.add_font("ArialUnicode", "", os.path.join(
            font_path, "arial-unicode-ms.ttf"), uni=True)
        pdf.add_font("ArialUnicode", "B", os.path.join(
            font_path, "arial-unicode-ms.ttf"), uni=True)
        pdf.add_font("ArialUnicode", "I", os.path.join(
            font_path, "arial-unicode-ms.ttf"), uni=True)
        pdf.add_font("ArialUnicode", "BI", os.path.join(
            font_path, "arial-unicode-ms.ttf"), uni=True)

        # Split the content by lines
        lines = toc_content.split("\n")
        for line in lines:
            if line.startswith("# "):  # Chapter Title
                pdf.set_font("ArialUnicode", style='B', size=16)
                pdf.cell(200, 10, txt=line[2:], ln=True)
            elif line.startswith("## "):  # Subsection Title
                pdf.set_font("ArialUnicode", style='B', size=14)
                pdf.cell(200, 10, txt=line[3:], ln=True)
            elif line.startswith("- "):  # Bullet Points
                pdf.set_font("ArialUnicode", size=12)
                pdf.cell(200, 10, txt=line, ln=True)
            else:  # Normal Text
                pdf.set_font("ArialUnicode", size=12)
                pdf.cell(200, 10, txt=line, ln=True)

        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Return as bytes
        return pdf_bytes


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

    elif option == "Fine-tuning":
        st.header("Upload your data here")
        uploaded_file = st.file_uploader(
            "Upload a file for fine-tuning", type=["txt", "pdf", "csv", "xlsx", "docx"])

        if uploaded_file is not None:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Fine-tuning in progress..."):
                assistant.finetune(file_path)
            st.success(
                "Fine-tuning done successfully. You can now chat with the updated RAG Assistant.")

    elif option == "Fine-tune with Google Drive":
        st.header("Fine-tune with Google Drive")
        folder_id = st.text_input("Enter Google Drive folder ID")

        if st.button("Fine-tune"):
            if folder_id:
                with st.spinner("Fine-tuning with Google Drive in progress..."):
                    message = assistant.finetune_gdrive(folder_id)
                st.success(message)

    elif option == "Generate Table of Contents":
        st.header("Generate Table of Contents")

        toc_prompt = st.text_area(
            "Enter your prompt for the Table of Contents:")
        if st.button("Generate"):
            if toc_prompt:
                with st.spinner("Generating Table of Contents..."):
                    toc_content = assistant.generate_toc(toc_prompt)
                    st.session_state.toc_content = toc_content
                st.success("Table of Contents generated successfully!")

        if st.session_state.toc_content:
            st.markdown("### Generated Table of Contents")
            st.markdown(st.session_state.toc_content)

            if st.button("Download as PDF"):
                pdf_bytes = assistant.save_toc_as_pdf(
                    st.session_state.toc_content)
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="table_of_contents.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()
