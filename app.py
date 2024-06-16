import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'
MAX_TOKENS = 512

# Load the locally downloaded model with GPU support
def load_llm():
    try:
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q4_0.bin",  # Updated model path
            model_type="llama",
            device="cuda",  # Use GPU
            max_new_tokens=512,  # Ensure new tokens do not exceed limit
            temperature=0.5
        )
        return llm
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def trim_query(query, max_tokens=MAX_TOKENS):
    # Tokenize the input query and trim it to fit within the maximum token limit
    query_tokens = query.split()
    if len(query_tokens) > max_tokens:
        return ' '.join(query_tokens[:max_tokens])
    return query

st.title("Llama Chatbot")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()
        
        if llm:
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

            def conversational_chat(query):
                # Trim the query to fit within the token limit
                trimmed_query = trim_query(query)
                result = chain({"question": trimmed_query, "chat_history": []})
                st.session_state['history'].append((query, result["answer"]))
                return result["answer"]
            
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = [f"Hello! Ask me anything about {uploaded_file.name} 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey! 👋"]
                
            # Container for the chat history
            response_container = st.container()
            # Container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                    submit_button = st.form_submit_button(label='Send')
                
                if submit_button and user_input:
                    output = conversational_chat(user_input)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
