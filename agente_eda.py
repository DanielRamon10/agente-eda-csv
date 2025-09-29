import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Agente EDA - Análise Genérica de CSV", layout="wide")

# Funções utilitárias
def analisar_tipos(df):
    return pd.DataFrame({
        'Tipo de Dado': df.dtypes,
        'Quantidade de Valores Nulos': df.isnull().sum(),
        'Valores Únicos': df.nunique()
    })

def conclusoes_automatica(df, memoria):
    conclusoes = []
    total_nulos = df.isnull().sum().sum()
    if total_nulos > 0:
        conclusoes.append(f"Existem {total_nulos} valores nulos no dataset, recomendando tratamento prévio à análise.")
    else:
        conclusoes.append("Não há valores nulos no dataset.")
    if 'Class' in df.columns:
        fraudes = int(df['Class'].sum())
        total = len(df)
        percent = fraudes / total * 100
        conclusoes.append(f"Foram identificadas {fraudes} transações fraudulentas ({percent:.4f}% do total), indicando dataset altamente desbalanceado.")
    if all([col.startswith('V') for col in df.columns if col.startswith('V')]):
        conclusoes.append("As colunas V1 a V28 passaram por redução de dimensionalidade (PCA), portanto não é possível saber seu significado real.")
    if 'Amount' in df.columns:
        q1 = df['Amount'].quantile(0.25)
        q3 = df['Amount'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df['Amount'] < (q1 - 1.5*iqr)) | (df['Amount'] > (q3 + 1.5*iqr))]
        if len(outliers) > 0:
            conclusoes.append(f"Foram detectados {len(outliers)} outliers na coluna 'Amount'. Recomenda-se avaliar seu impacto na análise.")
    memoria['conclusoes'] = conclusoes
    return conclusoes

if 'memoria' not in st.session_state:
    st.session_state['memoria'] = dict(conclusoes=[])

st.title("🧑‍💻 Agente Autônomo de EDA - Qualquer CSV!")

st.markdown("""
Este agente realiza análise exploratória automática de qualquer arquivo CSV e responde perguntas sobre os dados, gerando gráficos e conclusões.
""")

uploaded_file = st.file_uploader("Faça upload do seu arquivo CSV para iniciar a análise:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Prévia dos Dados")
    st.dataframe(df.head())

    st.subheader("Tipos de Dados e Informações Básicas")
    st.dataframe(analisar_tipos(df))

    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe().T)

    st.subheader("Visualização de Distribuição")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        col_hist = st.selectbox("Escolha uma coluna numérica para histograma:", num_cols)
        fig, ax = plt.subplots()
        df[col_hist].hist(bins=30, ax=ax)
        ax.set_title(f"Histograma de {col_hist}")
        st.pyplot(fig)
    with col2:
        col_box = st.selectbox("Escolha uma coluna numérica para boxplot:", num_cols, index=1 if len(num_cols)>1 else 0)
        fig2, ax2 = plt.subplots()
        ax2.boxplot(df[col_box].dropna())
        ax2.set_title(f"Boxplot de {col_box}")
        st.pyplot(fig2)

    st.subheader("Matriz de Correlação (variáveis numéricas)")
    corr = df[num_cols].corr()
    st.dataframe(corr)
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    cax = ax_corr.matshow(corr)
    fig_corr.colorbar(cax)
    ax_corr.set_xticks(range(len(num_cols)))
    ax_corr.set_yticks(range(len(num_cols)))
    ax_corr.set_xticklabels(num_cols, rotation=90, fontsize=8)
    ax_corr.set_yticklabels(num_cols, fontsize=8)
    st.pyplot(fig_corr)

    st.subheader("Gráfico de Dispersão entre Variáveis Numéricas")
    col_x = st.selectbox("Selecione X:", num_cols, index=0)
    col_y = st.selectbox("Selecione Y:", num_cols, index=1 if len(num_cols)>1 else 0)
    fig3, ax3 = plt.subplots()
    ax3.scatter(df[col_x], df[col_y], alpha=0.3)
    ax3.set_xlabel(col_x)
    ax3.set_ylabel(col_y)
    ax3.set_title(f"Scatter: {col_x} vs {col_y}")
    st.pyplot(fig3)

    st.subheader("Conclusões do Agente")
    conclusoes = conclusoes_automatica(df, st.session_state['memoria'])
    for c in conclusoes:
        st.write("- ", c)

    st.subheader("Pergunte ao agente sobre os dados:")
    pergunta = st.text_input("Digite sua pergunta (em português):")
    if pergunta:
        resposta = ""
        pergunta_lower = pergunta.lower()
        if "tipo" in pergunta_lower or "categorico" in pergunta_lower or "numérico" in pergunta_lower:
            resposta = "Tipos de dados das colunas:\n" + str(df.dtypes)
        elif "fraude" in pergunta_lower and "quantas" in pergunta_lower:
            if "Class" in df.columns:
                n_fraudes = int(df['Class'].sum())
                n_total = len(df)
                resposta = f"Existem {n_fraudes} fraudes, o que representa {(n_fraudes/n_total)*100:.4f}% das transações."
            else:
                resposta = "A coluna 'Class' não existe neste arquivo."
        elif "média" in pergunta_lower:
            medias = df.mean(numeric_only=True)
            resposta = f"Média das variáveis numéricas:\n{medias}"
        elif "mediana" in pergunta_lower:
            medianas = df.median(numeric_only=True)
            resposta = f"Mediana das variáveis numéricas:\n{medianas}"
        elif "máximo" in pergunta_lower or "maior valor" in pergunta_lower or "maximo" in pergunta_lower:
            maximos = df.max(numeric_only=True)
            resposta = f"Maior valor de cada coluna numérica:\n{maximos}"
        elif "mínimo" in pergunta_lower or "menor valor" in pergunta_lower or "minimo" in pergunta_lower:
            minimos = df.min(numeric_only=True)
            resposta = f"Menor valor de cada coluna numérica:\n{minimos}"
        elif "correlação" in pergunta_lower:
            resposta = "Matriz de correlação entre variáveis numéricas:\n" + str(df.corr())
        elif "outlier" in pergunta_lower or "valor atípico" in pergunta_lower:
            outlier_cols = []
            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))]
                if len(outliers) > 0:
                    outlier_cols.append(f"{col}: {len(outliers)} outliers")
            resposta = "Outliers encontrados:\n" + "\n".join(outlier_cols) if outlier_cols else "Nenhum outlier significativo encontrado."
        elif "conclusão" in pergunta_lower:
            resposta = "\n".join(st.session_state['memoria']['conclusoes'])
        else:
            resposta = "Pergunta não reconhecida diretamente. Tente perguntas sobre tipo de dado, média, fraudes, correlação, máximos/mínimos ou outliers."
        st.success(resposta)
else:
    st.info("Faça upload de um arquivo CSV para começar a análise.")

st.markdown("---")
st.caption("Desenvolvido para o Desafio Extra - I2A2 Academy 2025")
