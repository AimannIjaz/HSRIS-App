import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import pickle
import os
from collections import Counter

# ─── Load pre-computed data ───
@st.cache_resource
def load_data():
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsris_app_data.pkl')
    if not os.path.exists(pkl_path):
        st.error(f"Missing required file: {pkl_path}")
        st.stop()
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # Rebuild embedding layer
    emb = nn.Embedding(len(data['glove_tokens']), data['EMB_DIM'], padding_idx=0)
    emb.weight.data.copy_(torch.tensor(data['emb_matrix']))
    emb.weight.requires_grad = False
    data['embedding_layer'] = emb
    return data

data = load_data()
df = data['df']
vocab = data['vocab']
vocab2idx = data['vocab2idx']
idf = data['idf']
tfidf_dense = data['tfidf_dense']
glove_matrix = data['glove_matrix']
embedding_layer = data['embedding_layer']
glove_tok2idx = data['glove_tok2idx']
TOP_K = data['TOP_K']
EMB_DIM = data['EMB_DIM']
device = torch.device('cpu')

def tokenize(text):
    return re.findall(r'[a-z]+', str(text).lower())

def get_ngrams(tokens, n):
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tokenize_with_ngrams(text):
    tokens = tokenize(text)
    all_t = list(tokens)
    all_t.extend(get_ngrams(tokens, 2))
    all_t.extend(get_ngrams(tokens, 3))
    return all_t

def get_glove_indices(tokens):
    return [glove_tok2idx.get(t, 0) for t in tokens]

def get_sentence_vector(tokens, tfidf_row_dict):
    total_w = 0.0
    ws = torch.zeros(EMB_DIM)
    for tok, idx in zip(tokens, get_glove_indices(tokens)):
        if idx == 0: continue
        w = tfidf_row_dict.get(tok, 1e-5)
        ws += w * embedding_layer(torch.tensor([idx]))[0]
        total_w += w
    return ws / total_w if total_w > 0 else ws

def hybrid_search(query, alpha=0.4, top_k=5):
    q_toks = tokenize_with_ngrams(query)
    qt = Counter(q_toks)
    total = sum(qt.values()) or 1
    qv = torch.zeros(TOP_K)
    for tok, cnt in qt.items():
        if tok in vocab2idx:
            qv[vocab2idx[tok]] = (cnt/total) * idf[vocab2idx[tok]]
    qv = F.normalize(qv.unsqueeze(0), dim=1)
    
    q_uni = tokenize(query)
    rd = {vocab[j]: qv[0,j].item() for j in qv[0].nonzero(as_tuple=True)[0] if j.item() < len(vocab)}
    gv = get_sentence_vector(q_uni, rd)
    gv = F.normalize(gv.unsqueeze(0), dim=1)
    
    ts = torch.mm(qv, tfidf_dense.T).squeeze(0)
    gs = torch.mm(gv, glove_matrix.T).squeeze(0)
    fs = alpha * ts + (1-alpha) * gs
    top_i = fs.topk(top_k).indices.tolist()
    
    res = df.iloc[top_i][['Ticket Description','Ticket Subject','Ticket Type','Ticket Priority','Ticket Channel']].copy()
    res['Score'] = [fs[i].item() for i in top_i]
    return res, df.iloc[top_i]['Ticket Type'].mode()[0]

# ─── Streamlit UI ───
st.set_page_config(page_title="HSRIS", page_icon="🔬", layout="wide")
st.title("🔬 HSRIS — Hybrid Semantic Retrieval System")
st.markdown("Adjust **alpha** to shift between keyword matching (TF-IDF) and semantic matching (GloVe).")

col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_area("📝 Enter Ticket Description", height=120,
                         placeholder="e.g. I cannot login to my account, password reset not working...")
with col2:
    alpha = st.slider("⚖️ Alpha (α)", 0.0, 1.0, 0.4, 0.05,
                      help="0.0 = Pure GloVe (semantic) | 1.0 = Pure TF-IDF (keyword)")
    st.info(f"TF-IDF weight: {alpha:.0%}\\nGloVe weight: {1-alpha:.0%}")

if st.button("🔍 Search", type="primary") and query.strip():
    results, predicted_type = hybrid_search(query, alpha=alpha, top_k=3)
    st.success(f"**Predicted Ticket Type:** {predicted_type}")
    st.subheader("Top 3 Similar Past Resolutions")
    for i, (_, row) in enumerate(results.iterrows(), 1):
        with st.expander(f"Result {i} — [{row['Ticket Type']}] (Score: {row['Score']:.4f})", expanded=True):
            st.markdown(f"**Type:** {row['Ticket Type']} | **Priority:** {row['Ticket Priority']} | **Channel:** {row['Ticket Channel']}")
            st.markdown(f"**Subject:** {row['Ticket Subject']}")
            st.write(row['Ticket Description'])
