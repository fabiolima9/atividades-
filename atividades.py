import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

@st.cache_resource
def preparar_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

preparar_nltk()

st.set_page_config(page_title="Análise Avançada", page_icon="📈", layout="wide")
st.title("📈 Análise Avançada de Reclamações")

# Layout em colunas
col1, col2 = st.columns([1, 2])

with col1:
    texto = st.text_area("Cole as reclamações:", height=200)
    tipo_analise = st.radio("Tipo de Análise:", ["Palavras Simples", "Pares de Palavras (Bigramas)"])
    qtd = st.slider("Quantidade de termos:", 5, 50, 10)
    analisar = st.button("Analisar", type="primary")

if analisar and texto.strip():
    palavras = word_tokenize(texto.lower(), language='portuguese')
    stop_words_pt = set(stopwords.words('portuguese'))
    
    # Filtro de palavras úteis
    palavras_uteis = [p for p in palavras if p.isalpha() and p not in stop_words_pt]
    
    with col2:
        if palavras_uteis:
            if tipo_analise == "Pares de Palavras (Bigramas)":
                # Gera pares de palavras consecutivas
                termos = list(ngrams(palavras_uteis, 2))
                termos = [f"{w1} {w2}" for w1, w2 in termos]
            else:
                termos = palavras_uteis

            # Contagem
            contagem = Counter(termos)
            top_n = contagem.most_common(qtd)
            df = pd.DataFrame(top_n, columns=['Termo', 'Frequência'])
            
            st.subheader(f"Top {qtd} - {tipo_analise}")
            
            # Exibição do Gráfico e Tabela lado a lado
            c_grafico, c_tabela = st.columns(2)
            with c_grafico:
                st.bar_chart(df.set_index('Termo'))
            with c_tabela:
                st.dataframe(df, use_container_width=True)
            
            # Gerador de Nuvem de Palavras
            st.subheader("Nuvem de Palavras")
            texto_nuvem = " ".join(termos)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_nuvem)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
        else:
            st.warning("Texto insuficiente para análise.")