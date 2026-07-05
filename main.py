try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# from dotenv import load_dotenv
# load_dotenv()

from langchain_community.document_loaders import PyPDFLoader # 1. 문서 로더 (langchain_community 사용)
from langchain_text_splitters import RecursiveCharacterTextSplitter # 2. 텍스트 분할 (독립 패키지 사용)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # 3. OpenAI 모델 및 임베딩
from langchain_chroma import Chroma # 4. Chroma 벡터 저장소
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

#제목
st.title("ChatPDF")
st.write("---")


# 사이드바에서 OpenAI Key 입력 받기
with st.sidebar:
    st.title("설정")
    default_key = os.getenv("OPENAI_API_KEY", "")
    openai_key = st.text_input('OpenAI API Key', value=default_key, type='password', placeholder="sk-...")


#파일 업로드
uploaded_file = st.file_uploader("PDF파일을 올려주세요.", type = ['.pdf'])
st.write("---")

@st.cache_resource
def get_cached_vector_db(file_bytes, file_name, openai_api_key, chunk_size=1000, chunk_overlap=200):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, file_name)
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)
        
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
    
    # 텍스트 분할 (chunk_size를 1000으로 늘려 문맥이 끊기지 않도록 개선)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩 생성 및 Chroma DB 저장
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings_model)
    return db

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    if not openai_key:
        st.error("OpenAI API Key를 입력해주세요.")
        db = None
    else:
        # Streamlit 캐싱을 활용해 파일이 바뀌거나 재실행될 때 DB 중복 생성을 방지
        db = get_cached_vector_db(uploaded_file.getvalue(), uploaded_file.name, openai_key, chunk_size=1000, chunk_overlap=200)

if uploaded_file is not None and db is not None:

    #question
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")
    
    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
                
            if not question.strip():
                st.warning("질문을 입력하세요.")
            else:
                llm = ChatOpenAI(temperature=0, openai_api_key = openai_key, streaming = True, callbacks = [StreamingStdOutCallbackHandler()]) # 쓰고싶은 모델만 바꿔서 끼워넣으면 됩니다
                
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                docs = retriever.invoke(question)

                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"""제공된 Context를 최대한 활용하여 답변하세요. 만약 Context에 확실한 정보가 없다면 '주어진 문서에서 정보를 찾을 수 없습니다'라고 친절하게 답하고, 당신이 알고 있는 일반적인 지식은 답변에 포함하지 마세요.

Context:
{context}

Question: {question}"""

                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                print(answer)
                st.write(answer)
