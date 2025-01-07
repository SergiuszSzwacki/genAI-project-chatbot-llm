import json
import os
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.schema import Document  # Correct import for Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
import streamlit as st
import openai

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# Custom utilities (Ensure these are implemented in your 'lib' folder)
from lib import utils
from lib.streaming import StreamHandler

# Streamlit UI settings
st.set_page_config(page_title="Chat", page_icon="📄")
st.header('Chodkiewicz Chatbot')
st.write('Hi, I am Chodkiewicz Chad, happy to answer your questions about the Battle of Kircholm.')
#sidebar
st.sidebar.image("husar1.jpg", use_column_width=True)
st.sidebar.title("Chodkiewicz Chatbot")
st.sidebar.markdown("""
### About the Chatbot:
- This chatbot specializes in answering questions about the **Battle of Kircholm**.
- Powered by historical documents and advanced language models.
""")
class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    @st.spinner('Analyzing documents...')
    def import_source_documents(self):
        # Initialize document list
        docs = []
        files = []

        # Load documents from 'data' folder (both .txt and .json)
        for file in os.listdir("data"):
            file_path = os.path.join("data", file)

            try:
                if file.endswith(".txt"):
                    # For .txt files
                    with open(file_path, encoding='utf-8') as f:
                        docs.append(f.read())
                        files.append(file)
                elif file.endswith(".json"):
                    # For .json files
                    with open(file_path, encoding='utf-8') as f:
                        data = json.load(f)
                        docs.append(data.get('text', ''))  # Ensure 'text' is the key
                        files.append(file)
            except Exception as e:
                st.warning(f"Could not process file {file}: {e}")

        # Split documents and store in a vector database
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        # Store documents in the vector DB
        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)

        # System message prompt template
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with responding to questions based on attached websites content below. Never answer to any question that is not related to source documents.
            {context}
            
            Considering the text above, answer the following question. Only depend on source documents.
            No matter in what language is the question, always respond in English language.
            For every answer, provide a quote from the source document.
            {question}
            """
        )

        # Setup prompt and chain
        prompt = ChatPromptTemplate.from_messages([system_message_prompt])
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return qa_chain

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask for information from documents")

        if user_query:
            qa_chain = self.import_source_documents()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                result = qa_chain.invoke({"question": user_query}, {"callbacks": [st_cb]})
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                # To show references
                for doc in result['source_documents']:
                    filename = os.path.basename(doc.metadata['source'])
                    ref_title = f":blue[Source document: {filename}]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
