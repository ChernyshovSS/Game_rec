import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    df = pd.read_csv("~/Game_rec/data/game_data.csv")
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.to(device)
    return df, tokenizer, model

# Функция для создания эмбеддингов
def embed_bert_cls(text, model, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_mask)

        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()


def faiss_index(_df, _model, _tokenizer):
    _df["description"] = _df["description"].fillna("").astype(str)
    description_embeddings = np.vstack(
        _df["description"].apply(lambda x: embed_bert_cls(x, _model, _tokenizer)).values
    )
    index = faiss.IndexFlatL2(description_embeddings.shape[1])
    index.add(description_embeddings.astype(np.float32))
    return index, description_embeddings


def game_finder(query, df, model, tokenizer, index, description_embeddings, top_k=5):
    # Создание эмбеддингов для запроса
    query_embedding = embed_bert_cls(query, model, tokenizer).reshape(1, -1)

    # Поиск топ-K ближайших соседей с использованием Faiss
    _, top_k_indices = index.search(query_embedding.astype(np.float32), top_k)

    # Извлечение соответствующих строк из DataFrame
    result = df.iloc[top_k_indices[0]].copy()

    # Вычисление косинусного сходства для порядкового вывода
    similarities = cosine_similarity(
        query_embedding, description_embeddings[top_k_indices[0]]
    ).flatten()
    result["similarity"] = similarities

    # Сортировка по сходству
    return result.sort_values("similarity", ascending=False).head(top_k)


def show_random_games(df, num_games=10):
    random_games = df.sample(n=num_games)
    for _, row in random_games.iterrows():
        st.image(row['image_url'], width=150)
        st.markdown(f"### {row['title']}")
        st.markdown(f"{row['page_url']}")
        st.markdown(f"{row['description']}")
        st.markdown('---')