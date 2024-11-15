import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader, PyPDFLoader, WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from pinecone import Pinecone
from prompt import prompt_template_text, toc_prompt_template_text
from fpdf import FPDF
# from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


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
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1-aws')
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
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1-aws')
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

        assistant_response = chain.invoke(user_input)
        response_text = assistant_response['result']
        return response_text

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
    if "toc_content" not in st.session_state:
        st.session_state.toc_content = ""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    option = st.sidebar.selectbox(
        "Choose an option", ("Chat", "Fine-tuning", "Generate Table of Contents", "Web Content Q&A"))

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

    elif option == "Web Content Q&A":
        st.header("Web Content Q&A")

        # Input for web link
        url = st.text_input("Enter a web link to index:")

        if st.button("Index Content"):
            with st.spinner("Indexing content..."):
                # Load and process the web content
                loader = WebBaseLoader(url)
                docs = loader.load()

                # Split the documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(docs)

                # Create and store the vectorstore
                embeddings = OpenAIEmbeddings()
                # Pinecone(api_key="bf5d3307-78d9-4f61-8ced-b6d53a43e6c6",
                #          environment='us-east-1-aws')
                # st.session_state.vectorstore = PineconeVectorStore.from_documents(
                #     docs, embeddings, index_name="gerrylawchatbot")
                st.session_state.vectorstore = FAISS.from_documents(
                    documents=docs, embedding=embeddings)

            st.success("Content indexed successfully!")

        # Chat interface
        if st.session_state.vectorstore is not None:
            query = st.text_input("Ask a question about the indexed content:")

            if query:
                # Create the chain
                retriever = st.session_state.vectorstore.as_retriever()

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant for question-answering tasks. "
                     "Use the following pieces of retrieved context to answer the question. "
                     "If you don't know the answer, just say that you don't know. "
                     "keep the answer detailed.\n\n{context}"),
                    ("human", "{input}")
                ])

                combine_docs_chain = create_stuff_documents_chain(
                    assistant.llm, prompt)
                retrieval_chain = create_retrieval_chain(
                    retriever, combine_docs_chain)

                # Run the chain
                response = retrieval_chain.invoke({"input": query})

                st.write("Answer:", response['answer'])
        else:
            st.info("Please index a web page first.")


if __name__ == "__main__":
    main()
