from streamlit_elements import elements, mui, html
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from annotated_text import annotated_text
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import openai
import pinecone
from PIL import Image
import pickle
from pages.generate_result_img import *

@st.cache_resource(show_spinner=None)
def init_openai_key():
    openai.api_key = st.secrets.OPENAI_TOKEN

    return openai.api_key

with open('index_list.pickle', 'rb') as file:
    index_list = pickle.load(file)

def init_pinecone_connection():
    pinecone.init(
        api_key=st.secrets["PINECONE_KEY"],
        environment=st.secrets["PINECONE_REGION"]
    )
    pinecone_index = pinecone.Index('bookstore')
    return pinecone_index

@st.cache_data(show_spinner=None)
def generate_songs():
    df = pd.read_csv('./pages/data/melon_kakao_streamlit.csv')
    songs = df['song_name'] + ' | ' + df['artist_name_basket']

    return songs, df

def get_embedding(query):
    response = openai.Embedding.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def get_vectors_by_ids(pinecone_index, index_list):
    vector_data_list = []  # 벡터 데이터를 모을 리스트

    for s_id in index_list:
        # ID에 해당하는 벡터 데이터를 불러옴
        result = pinecone_index.retrieve(ids=[str(s_id)], namespace="playbooklist")

        # 결과에서 벡터 데이터 추출
        if result and result[0]['status'] == 'ok':
            vector_data = result[0]['data'][0]['vector']
            vector_data_list.append(vector_data)

    return vector_data_list

def _vector_search(query_embedding):
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
    )
    matches = results["matches"]
    return sorted([x["metadata"] for x in matches if x['metadata']['rating'] >= 8],
                  key=lambda x: (x['review_cnt'], x['rating']), reverse=True)[:5]

def generate_result():
    vector_data_list = get_vectors_by_ids(pinecone_index, index_list)
    index = [i for i in range(len(vector_data_list))]
    row_df = pd.DataFrame(vector_data_list, index=index)
    embedding_len = len(eval(row_df.loc[0, 'embedding']))
    embeddings = np.array([0.0 for x in range(embedding_len)])
    for embedding in list(row_df['embedding']):
        embeddings += eval(embedding)
    result = _vector_search(list(embeddings))
    return result

cur_img_index = 0  # cur_img_index를 전역 변수로 초기화
img_paths = []  # img_paths를 전역 변수로 초기화

def show_image():
    global cur_img_index, img_paths
    if not img_paths:  # 이미지 경로가 없을 때만 초기화
        cur_img_index = 0
        img_paths = []

        result = generate_result()
        mockup_img = generate_mockup_img()
        for index in range(len(result)):
            img_url = result[index]['img_url']
            title = result[index]['title']
            authors = result[index]['authors']
            # 결과 이미지를 result_0.png, result_1.png로 저장. 덮어쓰기해서 용량 아끼기 위함.
            generate_result_img(index, mockup_img, img_url, title, authors)

        if result:
            for i in range(len(result)):
                img_paths.append(f"./pages/result_img/result_{i}.png")

    return cur_img_index, img_paths

cur_img_index, img_paths = show_image()

if 'idx' not in st.session_state:
    st.session_state.idx = 0

def change():
    st.session_state.idx += 1
    if st.session_state.idx >= len(img_paths):
        st.session_state.idx = 0

def get_author_title(item):
    return f"**{item['authors']}** | **{item['publisher']}**"

if __name__ == '__main__':
    openai.api_key = init_openai_key()
    pinecone_index = init_pinecone_connection()

empty1, con1, empty2 = st.columns([0.2, 1.0, 0.2])
with empty1:
    st.empty()
with con1:
    with stylable_container(
            key="result_container",
            css_styles="""
            {
                border: 3px solid rgba(150, 55, 23, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
    ):
        c1, c2 = st.columns(2, gap="small")
        result = generate_result()
        mockup_img = generate_mockup_img()
        with c1:
            st.image(img_paths[st.session_state.idx])

            for index in range(len(result)):
                img_url = result[index]['img_url']
                title = result[index]['title']
                authors = result[index]['authors']
                # 결과 이미지를 result_0.png, result_1.png로 저장. 덮어쓰기해서 용량 아끼기 위함.
                generate_result_img(index, mockup_img, img_url, title, authors)

            next_img = st.button("**다음 장으로 ▶▶**")

            if next_img:
                change()

        with c2:
            want_to_main = st.button("새 플레이리스트 만들기 🔁")
            if want_to_main:
                switch_page("main")
            annotated_text(("**추천결과**", "", "#ff873d"))
            for i, item in enumerate(result):
                with st.expander(f"#{i + 1} {get_author_title(item)}"):
                    st.header(item["title"])
                    st.write(
                        f"**{item['authors']}** | {item['publisher']} | {item['published_at']} | [yes24]({item['url']})")
                    st.write(item["summary"])

with empty2:
    st.empty()
