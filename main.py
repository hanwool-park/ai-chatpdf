__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()

from langchain_community.document_loaders import PyPDFLoader # 1. 문서 로더 (langchain_community 사용)
from langchain_text_splitters import RecursiveCharacterTextSplitter # 2. 텍스트 분할 (독립 패키지 사용)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # 3. OpenAI 모델 및 임베딩
from langchain_chroma import Chroma # 4. Chroma 벡터 저장소
import streamlit as st
import tempfile
import os


#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF파일을 올려주세요.", type = ['.pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    #loader

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    
    #split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap = 150,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function = len,
        is_separator_regex= False,
    )

    texts = text_splitter.split_documents(pages)

    #embeddings
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #question
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")
    
    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
                
            if not question.strip():
                st.warning("질문을 입력하세요.")
            else:
                llm = ChatOpenAI(temperature=0) # 쓰고싶은 모델만 바꿔서 끼워넣으면 됩니다
                
                retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
                docs = retriever.invoke(question)

                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"""
    제공된 Context를 최대한 활용하여 답변하세요. 만약 Context에 확실한 정보가 없다면 '주어진 문서에서 정보를 찾을 수 없습니다'라고 친절하게 답하고, 당신이 알고 있는 일반적인 지식은 답변에 포함하지 마세요.
    
    Context:
    {context}

    Question: {question}
    
    """

                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                print(answer)
                st.write(answer)
