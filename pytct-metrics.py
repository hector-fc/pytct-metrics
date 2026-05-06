import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página deve ser o primeiro comando Streamlit
st.set_page_config(layout="wide", page_title="Análise Psicométrica TCT")

# ==========================================
# 1. MOTOR MATEMÁTICO (FUNÇÕES TCT)
# ==========================================

def gerar_matriz_sp(df_bruto):
    df_trabalho = df_bruto.copy()
    coluna_id = ['ID']
    colunas_questoes = [col for col in df_trabalho.columns if col not in coluna_id]
    
    df_trabalho['S'] = df_trabalho[colunas_questoes].sum(axis=1)
    vetor_p = df_trabalho[colunas_questoes].sum(axis=0)
    
    questoes_ordenadas = vetor_p.sort_values(ascending=False).index.tolist()
    df_ordenado = df_trabalho.sort_values(by='S', ascending=False).reset_index(drop=True)
    
    nova_ordem = coluna_id + questoes_ordenadas + ['S']
    df_final = df_ordenado[nova_ordem].copy()
    
    linha_dif = df_final[questoes_ordenadas].sum(axis=0).astype(object)
    linha_dif['ID'] = 'D'
    linha_dif['S'] = df_final['S'].sum()
    
    linha_media = df_final[questoes_ordenadas].mean(axis=0).astype(object)
    linha_media['ID'] = 'p*'
    linha_media['S'] = np.nan 
    
    linhas_resumo = pd.DataFrame([linha_dif, linha_media])
    return pd.concat([df_final, linhas_resumo], ignore_index=True)

def adicionar_indice_discriminacao(df_sp):
    df_trabalho = df_sp.copy()
    linhas_resumo = ['D', 'p*']
    df_alunos = df_trabalho[~df_trabalho['ID'].isin(linhas_resumo)].copy()
    
    colunas_questoes = [col for col in df_trabalho.columns if col not in ['ID', 'S']]
    vetor_S = df_alunos['S'].astype(float)
    
    linha_pbi = {'ID': 'r_pbi', 'S': np.nan}
    
    for item in colunas_questoes:
        X_i = df_alunos[item].astype(float)
        S_minus_i = vetor_S - X_i
        
        if X_i.std() == 0 or S_minus_i.std() == 0:
            r_pbi = np.nan 
        else:
            r_pbi = X_i.corr(S_minus_i)
        linha_pbi[item] = round(r_pbi, 3)
        
    df_linha_pbi = pd.DataFrame([linha_pbi])
    return pd.concat([df_trabalho, df_linha_pbi], ignore_index=True)

def adicionar_discriminacao_grupos_extremos(df_sp):
    df_trabalho = df_sp.copy()
    linhas_resumo = ['D', 'p*', 'r_pbi']
    df_alunos = df_trabalho[~df_trabalho['ID'].isin(linhas_resumo)].copy()
    df_alunos = df_alunos.sort_values(by='S', ascending=False).reset_index(drop=True)
    
    n = max(1, int(len(df_alunos) / 3))
    df_sup = df_alunos.head(n) 
    df_inf = df_alunos.tail(n) 
    
    colunas_questoes = [col for col in df_trabalho.columns if col not in ['ID', 'S']]
    linha_D = {'ID': 'D_i', 'S': np.nan}
    
    for item in colunas_questoes:
        linha_D[item] = round((df_sup[item].sum() - df_inf[item].sum()) / n, 3)
        
    df_linha_D = pd.DataFrame([linha_D])
    return pd.concat([df_trabalho, df_linha_D], ignore_index=True)

def calcular_mci_expandido(df_sp):
    df_final = df_sp.copy()
    linhas_resumo = ['D', 'p*', 'r_pbi', 'D_i']
    df_alunos = df_final[~df_final['ID'].isin(linhas_resumo)].copy()
    
    colunas_itens = [col for col in df_alunos.columns if col not in ['ID', 'S']]
    d_i = df_final.loc[df_final['ID'] == 'D', colunas_itens].values.flatten().astype(float)
    d_i = np.sort(d_i)[::-1] 
    L = len(colunas_itens)
    
    indices_cautela = []
    lista_chutes = []
    
    for idx, row in df_alunos.iterrows():
        s_n = int(row['S'])
        x_ni = row[colunas_itens].values.astype(int)
        
        chutes_aluno = np.sum(x_ni[s_n:])
        lista_chutes.append(int(chutes_aluno))
        
        if s_n == 0 or s_n == L:
            indices_cautela.append(0.0)
            continue
            
        numerador = np.sum((1 - x_ni[:s_n]) * d_i[:s_n]) + np.sum(x_ni[s_n:] * d_i[s_n:])
        c_n_star = numerador / sum(d_i) if sum(d_i) > 0 else 0.0
        indices_cautela.append(round(c_n_star, 3))

    df_final['chutes'] = df_final.index.map(dict(zip(df_alunos.index, lista_chutes)))
    df_final['C_n'] = df_final.index.map(dict(zip(df_alunos.index, indices_cautela)))
    
    df_final.loc[df_final['ID'].isin(linhas_resumo), ['chutes', 'C_n']] = np.nan
    return df_final

def calcular_cronbach(df):
    L = df.shape[1]
    if L <= 1: return 0.0
    variancias_itens = df.var(ddof=1).sum()
    var_total = df.sum(axis=1).var(ddof=1)
    if var_total == 0: return 0.0
    return (L / (L - 1)) * (1 - (variancias_itens / var_total))

# ==========================================
# 2. MOTOR DE RENDERIZAÇÃO (ESTILOS)
# ==========================================

def colorir_matriz_sp(linha):
    """ Pinta o fundo das células indicando as 4 zonas de resposta. """
    estilos = [''] * len(linha)
    linhas_resumo = ['D', 'p*', 'r_pbi', 'D_i']
    
    if pd.isna(linha.get('S')) or linha.get('ID') in linhas_resumo:
        return estilos

    escore_aluno = int(linha['S'])
    ordem_questao = 1 
    
    for i, coluna in enumerate(linha.index):
        if coluna not in ['ID', 'S', 'C_n', 'chutes']:
            valor = linha[coluna]
            if ordem_questao <= escore_aluno:
                estilos[i] = 'background-color: lightgreen; color: black' if valor == 1 else 'background-color: lightcoral; color: black'
            else:
                estilos[i] = 'background-color: gold; color: black' if valor == 1 else 'background-color: whitesmoke; color: gray'
            ordem_questao += 1    
                    
    return estilos

def colorir_apenas_curvas_sp(linha, dict_dif, itens):
    """ Pinta apenas as fronteiras das Curvas Student (S) e Problem (P). """
    estilos = [''] * len(linha)    
    linhas_resumo = ['D', 'p*', 'r_pbi', 'D_i']
    
    if pd.isna(linha.get('S')) or linha.get('ID') in linhas_resumo:
        return estilos

    escore = int(linha['S'])
    rank = linha.name + 1  
    posicao_item = 1 
    
    for i, col in enumerate(linha.index):
        if col in itens:
            css = ""            
            is_curva_s = (posicao_item == escore)
            dificuldade_item = int(dict_dif.get(col, 0))
            is_curva_p = (rank == dificuldade_item)
            
            if is_curva_s and is_curva_p:
                css = "background-color: #32CD32; color: white; font-weight: bold;" 
            elif is_curva_s:
                css = "background-color: lightgreen; color: black;"
            elif is_curva_p:
                css = "background-color: gold; color: black;"
            
            estilos[i] = css
            posicao_item += 1 
            
    return estilos

# ==========================================
# 3. INTERFACE E FLUXO DE DADOS
# ==========================================

st.title("📊 Análise de Teste pelo Método TCT e Curva S-P")

st.sidebar.header("📥 Importar Dados")
uploaded_file = st.sidebar.file_uploader("Carregar matriz de respostas (.csv)", type="csv")

# Seleção da Base de Dados
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    st.sidebar.success("Arquivo carregado com sucesso!")
else:
    st.info("📊 Exibindo dados de exemplo. Carregue seu próprio arquivo CSV no menu lateral para analisar a sua turma.")
    data = {
        'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'i1': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'i2': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'i3': [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        'i4': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        'i5': [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)

Nlin,Lcol =  df.shape 

# Processamento do Pipeline Matemático
matriz_sp = gerar_matriz_sp(df)
matriz_com_disc = adicionar_indice_discriminacao(matriz_sp)
matriz_completa = adicionar_discriminacao_grupos_extremos(matriz_com_disc)
matriz_final = calcular_mci_expandido(matriz_completa)

# Cálculo de KPIs Globais
questoes = [col for col in df.columns if col != 'ID']
df_escores = df[questoes].sum(axis=1)
alfa = calcular_cronbach(df[questoes])

col1, col2, col3 = st.columns(3)
col1.metric("Alfa de Cronbach", f"{alfa:.2f}", help="Confiabilidade interna da prova")
col2.metric("Média de Acertos", f"{df_escores.mean():.1f}")
col3.metric("Total de Examinandos", len(df))

st.divider()

# ==========================================
# 4. EXIBIÇÃO DOS RESULTADOS VISUAIS
# ==========================================

# Extração do dicionário de dificuldades (Executado apenas uma vez)
linha_dif = matriz_final[matriz_final['ID'] == 'D']
dict_dificuldades = linha_dif[questoes].iloc[0].to_dict() if not linha_dif.empty else {}

# Tabela 1: Curvas S-P
st.subheader("1. Dispersão das Curvas S-P")
st.write("🟩 Curva Student (S) | 🟨 Curva Problem (P) | 🟢 Intersecção")


# 1. Defina o tamanho do seu recorte (ex: Nlin = 5 alunos)
# Nlin = 5 

# 2. Fatie o DataFrame ORIGINAL (matriz_final) ANTES de aplicar o estilo
matriz_fatiada = matriz_final.iloc[0:Nlin+1, 0:Lcol+1]

# 3. Aplique o estilo (formatação e cores) apenas na fatia selecionada
styled_curvasP1 = matriz_fatiada.style.format(precision=2).apply(
    lambda row: colorir_apenas_curvas_sp(row, dict_dificuldades, questoes), axis=1
)

# 4. Renderize no Streamlit
st.dataframe(styled_curvasP1, use_container_width=True)

# styled_curvas = matriz_final.style.format(precision=2).apply(
#    lambda row: colorir_apenas_curvas_sp(row, dict_dificuldades, questoes), axis=1
# )

# styled_curvasP1 = styled_curvas.iloc[0:Nlin+2,:].style    
# st.dataframe(styled_curvasP1, use_container_width=True)




# Tabela 2: Matriz Completa
st.subheader("2. Matriz de Respostas Zonal")
st.write("🟩 Acerto Esperado | 🟥 Erro Anômalo | 🟨 Acerto Inesperado (Chute) | ⬜ Erro Esperado")

colunas_matriz = ['ID'] + questoes + ['S', 'chutes', 'C_n']
colunas_ordenadas = [c for c in matriz_final.columns if c in colunas_matriz]

styled_matriz = matriz_final[colunas_ordenadas].style.format(precision=2).apply(colorir_matriz_sp, axis=1)
st.dataframe(styled_matriz, use_container_width=True)

st.divider()

col_a, col_b = st.columns(2)

# Tabela 3: Alunos
with col_a:
    st.subheader("3. Análise dos Examinandos")
    df_alunos_exibicao = matriz_final[~matriz_final['ID'].isin(['D', 'p*', 'r_pbi', 'D_i'])].copy()
    df_alunos_exibicao['chutes'] = df_alunos_exibicao['chutes'].astype(int)
    df_alunos_exibicao['S'] = df_alunos_exibicao['S'].astype(int)
    
    st.dataframe(
        df_alunos_exibicao[['ID', 'chutes', 'S', 'C_n']].style.format({"C_n": "{:.2f}"}), 
        use_container_width=True, hide_index=True
    )

# Tabela 4: Itens
with col_b:
    st.subheader("4. Análise dos Itens")
    df_resumo = matriz_final[matriz_final['ID'].isin(['D', 'p*', 'r_pbi', 'D_i'])].copy()
    df_resumo.set_index('ID', inplace=True)
    
    df_itens = df_resumo[questoes].T.reset_index().rename(columns={'index': 'i'})
    df_itens = df_itens[['i', 'D', 'p*', 'r_pbi', 'D_i']]
    
    st.dataframe(
        df_itens.style.format({"D": "{:.0f}", "p*": "{:.2f}", "r_pbi": "{:.2f}", "D_i": "{:.2f}"}), 
        use_container_width=True, hide_index=True
    )

st.divider()

# Nomenclatura
st.subheader("📚 Nomenclatura das Variáveis")
st.markdown("""
* **i**: Identificador do Item (Questão).
* **ID**: Identificador do Examinando (Aluno).
* **S**: Escore bruto do examinando (Total de acertos).
* **chutes**: Quantidade de acertos do aluno na zona de desconhecimento.
* **C_n**: Índice de Cautela Global.
* **D**: Dificuldade bruta do item.
* **p\***: Média de acertos do item.
* **r_pbi**: Correlação ponto-bisserial corrigida.
* **D_i**: Índice de discriminação por grupos extremos.
""")


