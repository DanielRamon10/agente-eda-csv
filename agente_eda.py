import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Agente EDA - Análise Genérica de CSV", layout="wide")

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
        # Tipos de dados
        if "tipo" in pergunta_lower or "categorico" in pergunta_lower or "numérico" in pergunta_lower:
            resposta = "Tipos de dados das colunas:\n" + str(df.dtypes)
        # Fraudes
        elif ("fraude" in pergunta_lower or "fraudulenta" in pergunta_lower) and ("quantas" in pergunta_lower or "percent" in pergunta_lower or "proporç" in pergunta_lower or "porcent" in pergunta_lower):
            if "Class" in df.columns:
                n_fraudes = int(df['Class'].sum())
                n_total = len(df)
                perc_fraude = n_fraudes / n_total * 100
                resposta = f"Existem {n_fraudes} fraudes ({perc_fraude:.4f}%) no total de {n_total} transações."
            else:
                resposta = "A coluna 'Class' não existe neste arquivo."
        # Média
        elif "média" in pergunta_lower or "media" in pergunta_lower:
            medias = df.mean(numeric_only=True)
            resposta = f"Média das variáveis numéricas:\n{medias}"
        # Mediana
        elif "mediana" in pergunta_lower:
            medianas = df.median(numeric_only=True)
            resposta = f"Mediana das variáveis numéricas:\n{medianas}"
        # Moda
        elif "moda" in pergunta_lower or "valor mais comum" in pergunta_lower or "mais frequente" in pergunta_lower:
            modas = df.mode(numeric_only=True).iloc[0]
            resposta = f"Moda (valor mais frequente) das variáveis numéricas:\n{modas}"
        # Desvio padrão
        elif "desvio padrão" in pergunta_lower:
            stds = df.std(numeric_only=True)
            resposta = f"Desvio padrão das variáveis numéricas:\n{stds}"
        # Variância
        elif "variância" in pergunta_lower or "variancia" in pergunta_lower:
            vars = df.var(numeric_only=True)
            resposta = f"Variância das variáveis numéricas:\n{vars}"
        # Máximos
        elif "máximo" in pergunta_lower or "maior valor" in pergunta_lower or "maximo" in pergunta_lower:
            maximos = df.max(numeric_only=True)
            resposta = f"Maior valor de cada coluna numérica:\n{maximos}"
        # Mínimos
        elif "mínimo" in pergunta_lower or "menor valor" in pergunta_lower or "minimo" in pergunta_lower:
            minimos = df.min(numeric_only=True)
            resposta = f"Menor valor de cada coluna numérica:\n{minimos}"
        # Soma
        elif "soma" in pergunta_lower:
            somas = df.sum(numeric_only=True)
            resposta = f"Soma dos valores de cada coluna numérica:\n{somas}"
        # Nulos
        elif "nulo" in pergunta_lower or "faltante" in pergunta_lower:
            nulos = df.isnull().sum()
            resposta = f"Quantidade de valores nulos por coluna:\n{nulos}"
        # Linhas e colunas
        elif "quantos registros" in pergunta_lower or "quantas linhas" in pergunta_lower or "quantas colunas" in pergunta_lower:
            resposta = f"O arquivo possui {df.shape[0]} linhas e {df.shape[1]} colunas."
        # Colunas do arquivo
        elif "quais as colunas" in pergunta_lower or "nome das colunas" in pergunta_lower:
            resposta = f"As colunas do arquivo são: {list(df.columns)}"
        # Valor mais comum em uma coluna específica
        elif "valor mais comum" in pergunta_lower or "valor mais frequente" in pergunta_lower:
            # Tentativa de identificar coluna
            for col in df.columns:
                if col.lower() in pergunta_lower:
                    valor = df[col].mode().iloc[0]
                    resposta = f"O valor mais comum na coluna {col} é: {valor}"
                    break
        # Porcentagem/proporção das classes (exemplo: Class)
        elif "porcent" in pergunta_lower or "proporç" in pergunta_lower or "percent" in pergunta_lower:
            if "Class" in df.columns and set(df["Class"].unique()) == {0,1}:
                n_fraude = int(df["Class"].sum())
                n_total = len(df)
                perc_fraude = n_fraude / n_total * 100
                perc_normal = 100 - perc_fraude
                resposta = (
                    f"Porcentagem de transações normais: {perc_normal:.4f}%\n"
                    f"Porcentagem de transações fraudulentas: {perc_fraude:.4f}%"
                )
            else:
                cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                if cat_cols:
                    for col in cat_cols:
                        proporcoes = df[col].value_counts(normalize=True) * 100
                        resposta += f"Proporção por categoria em {col}:\n{proporcoes}\n"
                else:
                    resposta = "Nenhuma coluna categórica encontrada para calcular porcentagem."
        # Outliers
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
        # Intervalo dos valores
        elif "intervalo" in pergunta_lower:
            intervalo = df[num_cols].agg(['min', 'max'])
            resposta = f"Intervalo de valores (mínimo e máximo):\n{intervalo}"
        # Distribuição dos valores (mostrar histograma)
        elif "distribuição" in pergunta_lower or "histograma" in pergunta_lower:
            for col in num_cols:
                if col.lower() in pergunta_lower:
                    fig, ax = plt.subplots()
                    df[col].hist(bins=30, ax=ax)
                    ax.set_title(f"Histograma de {col}")
                    st.pyplot(fig)
                    resposta = f"Distribuição da coluna {col} exibida acima."
                    break
            else:
                resposta = "Especifique o nome de uma coluna numérica para mostrar a distribuição/histograma."
        # Top N valores (maiores/menores)
        elif "maiores" in pergunta_lower or "top" in pergunta_lower:
            for col in num_cols:
                if col.lower() in pergunta_lower:
                    topn = df[col].nlargest(10)
                    resposta = f"Top 10 maiores valores em {col}:\n{topn}"
                    break
        elif "menores" in pergunta_lower:
            for col in num_cols:
                if col.lower() in pergunta_lower:
                    botn = df[col].nsmallest(10)
                    resposta = f"Top 10 menores valores em {col}:\n{botn}"
                    break
        # Correlação entre duas colunas
        elif "correlação" in pergunta_lower or "correlacao" in pergunta_lower or "correlation" in pergunta_lower:
            found_cols = []
            for col in num_cols:
                if col.lower() in pergunta_lower:
                    found_cols.append(col)
            if len(found_cols) == 2:
                corr_val = df[found_cols[0]].corr(df[found_cols[1]])
                resposta = f"Correlação entre {found_cols[0]} e {found_cols[1]}: {corr_val:.4f}"
            else:
                resposta = "Informe os nomes de duas colunas numéricas para calcular a correlação."
        # Quantidade de categorias em uma coluna
        elif "categorias" in pergunta_lower:
            for col in df.columns:
                if col.lower() in pergunta_lower:
                    ncat = df[col].nunique()
                    resposta = f"A coluna {col} possui {ncat} categorias únicas."
                    break
        # Conclusões do agente
        elif "conclusão" in pergunta_lower or "conclusao" in pergunta_lower:
            resposta = "\n".join(st.session_state['memoria']['conclusoes'])
        else:
            resposta = "Pergunta não reconhecida diretamente. Tente perguntas sobre tipo de dado, média, fraudes, máximos/mínimos, proporções, outliers, intervalos, distribuição ou colunas."
        st.success(resposta)
else:
    st.info("Faça upload de um arquivo CSV para começar a análise.")

st.markdown("---")
st.caption("Desenvolvido para o Desafio Extra - I2A2 Academy 2025")
