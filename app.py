import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from model import load_data, embed_bert_cls, faiss_index, game_finder, show_random_games


st.set_page_config(
    page_title="Поиск игр по описанию",
    page_icon=":video_game:",
    layout="wide",
)

# Загружаем данные и модель
df, tokenizer, model = load_data()
index, description_embeddings = faiss_index(df, model, tokenizer)

# Создаем боковую панель
st.sidebar.title("Навигация")

# Выбор страницы
page = st.sidebar.radio("Выберите страницу", ["Стартовая","Случайные игры", "Поиск по описанию"])

# Страница "Стартовая"
if page == "Стартовая":
    st.title("Добро пожаловать в поисковик игр!")
    st.markdown("## Информация о приложении:")
    st.markdown(f"- Количество игр: 2000")
    st.markdown(f"- Источник парсинга: [IWANTGAMES](https://iwant.games/games/)")
    st.markdown(f"- Время парсинга: 22 минуты")
    st.markdown(f"- Модель: [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2/blob/main/README.md)")


# Страница "Случайные игры"
if page == "Случайные игры":
    st.title("Могу порекомендовать игры")
    if st.button("Во что поиграть?"):
    	show_random_games(df)


# Страница "Поиск"
elif page == "Поиск по описанию":
    st.title("Поиск игр по описанию")

    query = st.text_area("Введите описание игры", height=512)

    top_k = st.slider("Количество результатов", 1, 20, 5)

    if st.button("Найти"):
        if query:
            results = game_finder(
                query,
                df,
                model,
                tokenizer,
                index,
                description_embeddings,
                top_k=top_k,
            )
            st.write('Найденные игры:')
            for _, row in results.iterrows():
                st.image(row['image_url'], width=150)
                st.markdown(f"### {row['title']}")
                st.markdown(f"{row['page_url']}")
                st.markdown(f"{row['description']}")
                st.markdown(f"Косинусное сходство: {row['similarity']:.4f}")
                st.markdown('---')
        else:
            st.write('Введите описание для поиска.')

