import getpass
import os

import streamlit as st
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model # Assuming this is a custom or local utility function
from cl import get_streamlit_cb

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Constants
CHROMA_DB_DIR = "./chroma_db"
HTML_DATA_DIR = "./html_data"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
LLM_PROVIDER = "openai"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def setup_api_key():
    """Ensure OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass(
            "Enter API key for OpenAI: "
        )


def load_documents(data_dir):
    """Load HTML documents from the specified directory."""
    loader = DirectoryLoader(
        data_dir, glob="**/*.html", show_progress=True, recursive=True
    )
    return loader.load()


def split_documents(docs):
    """Split documents into chunks."""
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     add_start_index=True,
    # )
    # return text_splitter.split_documents(docs)
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai.embeddings import OpenAIEmbeddings
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    return text_splitter.split_documents(docs)


def load_or_create_vectorstore(persist_dir, embeddings):
    """Load vector store from disk or create it if it doesn't exist."""
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Loading existing vector store from {persist_dir}")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("Creating new vector store...")
        docs = load_documents(HTML_DATA_DIR)
        if not docs:
             st.error(f"No documents found in {HTML_DATA_DIR}. Please add HTML files.")
             st.stop()
        print(f"Loaded {len(docs)} documents.")
        all_splits = split_documents(docs)
        print(f"Split documents into {len(all_splits)} chunks.")
        if not all_splits:
             st.error("Failed to split documents into chunks.")
             st.stop()
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=embeddings, persist_directory=persist_dir
        )
        print(f"Created and persisted vector store to {persist_dir}")
    return vectorstore


def build_rag_chain(retriever, llm):
    """Build the RAG chain."""
    prompt_hub = hub.pull("rlm/rag-prompt") # This prompt is more generic. Let's stick to the custom one for specific instructions.

    system_prompt = (
"""
"You are an assistant for question-answering tasks. "
"Use the following pieces of retrieved context to answer "
"the question. If you don't know the answer, say that you "
"don't know. Use three sentences maximum and keep the "
"answer concise."
"Ты чат бот по недвижимости. Отвечай на вопросы по недвижимости. "
"Если ты считаешь, что нужно дополнительно сходить в базу данных, скажи "
"Вот примеры вопросов из FAQ и ответы на них:"
1. Какие гарантии на вложения?
Нашей гарантией результата являются уже построенные комплексы вилл и таунхаусов.

- Их проверенные фин модели и уникальные локации.
- Мы точно знаем сколько инвестор заработает на стройке и на аренде.
- Используем систему безопасных платежей на эскроу счет.
- Также мы даем самую большую гарантию среди застройщиков на качество стройки:
- 1 год на все общие дефекты
- 5 лет на скрытые дефекты в структуре и несущие конструкции
- 7 лет на гидроизоляцию и анти термитную обработку

2. Обязательно ли прилетать на Бали, чтобы купить недвижимость?
- Нет. По законам Индонезии сделку можно провести дистанционно.
- Дополнительных расходов это не подразумевает.

3. Какие дополнительные расходы?
- Итоговая стоимость сделки: стоимость виллы + 1% услуги нотариуса.
- В стоимость включено все под ключ: земля, недвижимость, мебель, налоги.

4. Какие документы я получу на руки?
- Вы всегда получаете прямое право владения землей, зданием, все лицензии, сертификаты.
- Мы не используем сомнительных форм собственности и хитростей с договорами.
- Инвестор имеет исключительные юридические права на весь объект покупки.

5. Сколько длится стройка?
- Срок стройки любого комплекса строго регламентирован договором – 12 месяцев.
- Также в договоре заложена гарантия выполнения обязательств застройщиком.

6. Как я получу свой доход?
Перечисление дохода клиенту происходит 1 раз в месяц по заранее согласованному каналу:

- переводом рупиями на счет в Индонезии, который мы можем помочь открыть
- наличными в офисе (доллары, евро, рупии)
- swift переводом
- криптой
- переводом $ в СШA на счет в США
- переводом € на евровый счет внутри еврозоны"
Контекст: {context}
"""
)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# --- Main Streamlit App Logic ---

st.set_page_config(page_title="Real Estate Chatbot", layout="wide")
st.title("BREIG Helper")

# Setup API key
setup_api_key()

# Initialize models
try:
    llm = init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
except Exception as e:
    st.error(f"Failed to initialize language models: {e}")
    st.stop()


# Load or create vector store
try:
    vectorstore = load_or_create_vectorstore(CHROMA_DB_DIR, embeddings)
except Exception as e:
    st.error(f"Failed to load or create vector store: {e}")
    st.stop()

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Build RAG chain
rag_chain = build_rag_chain(retriever, llm)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("Показанные документы"):
                for doc in message["context"]:
                     st.write(f"**Источник:** {doc.metadata.get('source', 'N/A')}")
                     st.caption(doc.page_content)
                     st.divider()


# React to user input
if user_input := st.chat_input("Задайте вопрос о недвижимости..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = rag_chain.invoke({"input": user_input},
                        config={"callbacks" :[get_streamlit_cb(st.empty())]})
            answer = response["answer"]
            context_docs = response["context"]

            st.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer, "context": context_docs})

            # Optionally display context documents used for the answer
            with st.expander("Показанные документы"):
                 for doc in context_docs:
                     st.write(f"**Источник:** {doc.metadata.get('source', 'N/A')}")
                     st.caption(doc.page_content)
                     st.divider()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Ошибка: {e}"})
