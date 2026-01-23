import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================================================================================
#                           CONFIGURAÇÃO INTELIGENTE DE CAMINHOS
# ==================================================================================

# 1. Define a pasta onde o script está rodando
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

# 2. Define onde deveriam estar os arquivos Excel (ANOS INICIAIS)
CAMINHO_BNCC = os.path.join(DIRETORIO_ATUAL, "bncc_df_anosiniciais.xlsx")
CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "curriculo_anos_iniciais_orientado_reescrito_v2.xlsx")

# Fallback de nome de arquivo caso tenha mudado
if not os.path.exists(CAMINHO_CURRICULO):
    CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "curriculo_anos_iniciais_orientacoes.xlsx")

# 3. Função para CAÇAR o modelo dentro da estrutura complexa de pastas
def encontrar_modelo_real(pasta_base):
    print(f"🔍 Rastreando modelo dentro de: {pasta_base}")
    
    if not os.path.exists(pasta_base):
        print(f"⚠️ Pasta 'models' não encontrada em {pasta_base}")
        return None

    # Procura pelo arquivo 'config.json' recursivamente
    for root, dirs, files in os.walk(pasta_base):
        if 'config.json' in files and ('pytorch_model.bin' in files or 'model.safetensors' in files):
            print(f"🎯 Modelo encontrado em: {root}")
            return root
            
    print("❌ Arquivos do modelo (config.json) não encontrados na estrutura.")
    return None

# Procura na pasta 'models' dentro do diretório atual
PASTA_MODELS_GERAL = os.path.join(DIRETORIO_ATUAL, 'models')
CAMINHO_FINAL_MODELO = encontrar_modelo_real(PASTA_MODELS_GERAL)

# Configuração de fallback
if CAMINHO_FINAL_MODELO is None:
    print("⚠️ Usando configuração de fallback (cache folder)...")
    CAMINHO_FINAL_MODELO = 'all-MiniLM-L6-v2'
    CACHE_FOLDER = PASTA_MODELS_GERAL
else:
    CACHE_FOLDER = None

# ==================================================================================
#                           CONFIGURAÇÕES DA ANÁLISE
# ==================================================================================
CONFIGURACOES = {
    'NOTA_CORTE': 0.80,
    'ARQUIVO_BNCC': CAMINHO_BNCC,
    'ARQUIVO_CURRICULO': CAMINHO_CURRICULO,
    'MODELO_EMBEDDINGS': CAMINHO_FINAL_MODELO,
    'MOSTRAR_TOP_MATCHES_TERMINAL': 10,
    'MOSTRAR_TOP_CORRESPONDENCIAS': 15,
    'PREFIXO_ARQUIVOS': f"corte_{int(0.80*100)}pct_",
    'GERAR_HEATMAP': True,
    'TAMANHO_HEATMAP': (20, 20),
}

CONFIGURACOES['PREFIXO_ARQUIVOS'] = f"corte_{int(CONFIGURACOES['NOTA_CORTE']*100)}pct_"

print("="*80)
print("🚀 ANALISADOR DE SIMILARIDADE - ANOS INICIAIS (OFFLINE/LOCAL)")
print("="*80)
print(f"📂 Diretório: {DIRETORIO_ATUAL}")
print(f"🤖 Modelo: {os.path.basename(str(CAMINHO_FINAL_MODELO))}")
print(f"📊 BNCC: {os.path.basename(CONFIGURACOES['ARQUIVO_BNCC'])}")
print(f"📊 Currículo: {os.path.basename(CONFIGURACOES['ARQUIVO_CURRICULO'])}")
print("="*80)

# ==================================================================================
#                           PROCESSAMENTO DOS DADOS
# ==================================================================================

# Carregamento dos dados
try:
    if not os.path.exists(CONFIGURACOES['ARQUIVO_BNCC']):
        raise FileNotFoundError(f"Arquivo BNCC não encontrado: {CONFIGURACOES['ARQUIVO_BNCC']}")
    if not os.path.exists(CONFIGURACOES['ARQUIVO_CURRICULO']):
        raise FileNotFoundError(f"Arquivo Currículo não encontrado: {CONFIGURACOES['ARQUIVO_CURRICULO']}")

    bncc_df_inf = pd.read_excel(CONFIGURACOES['ARQUIVO_BNCC'])
    curriculo_df_inf = pd.read_excel(CONFIGURACOES['ARQUIVO_CURRICULO'])
    print(f"✅ Dados carregados: {len(bncc_df_inf)} habilidades BNCC, {len(curriculo_df_inf)} habilidades currículo")
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    raise

# Verificação das colunas necessárias nos DataFrames
def checar_colunas_bncc(df):
    colunas_necessarias = ['EIXO', 'HABILIDADE', 'EXEMPLOS']
    for col in colunas_necessarias:
        assert col in df.columns, f"BNCC deve ter a coluna '{col}'"
    print("✅ Colunas verificadas na BNCC")

def checar_colunas_curriculo(df):
    colunas_necessarias = ['DISCIPLINA', 'HABILIDADES', 'ORIENTACOES_PEDAGOGICAS']
    for col in colunas_necessarias:
        assert col in df.columns, f"Currículo deve ter a coluna '{col}'"
    print("✅ Colunas verificadas no Currículo")

checar_colunas_bncc(bncc_df_inf)
checar_colunas_curriculo(curriculo_df_inf)

# Normaliza os nomes das colunas para evitar problemas de espaço/acentuação
curriculo_df_inf.columns = curriculo_df_inf.columns.str.strip()
bncc_df_inf.columns = bncc_df_inf.columns.str.strip()

# ==================================================================================
#                    FUNÇÃO DE EXTRAÇÃO DE CÓDIGOS
# ==================================================================================

def extrair_codigo(obj):
    if pd.isna(obj): return "(SEM_COD)"
    obj_str = str(obj)
    patterns = [
        r'\(([A-Z]{2}\d{2}[A-Z]{2}\d{2})\)',
        r'\(([A-Z]+\d+[A-Z]*\d*)\)',
        r'\(([A-Z0-9]+)\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, obj_str)
        if match: return f"({match.group(1)})"
    match_inicio = re.search(r'^\(([^)]+)\)', obj_str)
    if match_inicio: return f"({match_inicio.group(1)})"
    return f"({obj_str[:15]}...)" if len(obj_str) > 15 else f"({obj_str})"

# Função para concatenar as features relevantes
def concat_features_bncc(df):
    return (df['EIXO'].astype(str) + " | " + df['HABILIDADE'].astype(str) + " | " + df['EXEMPLOS'].astype(str))

def concat_features_curriculo(df):
    return (df['DISCIPLINA'].astype(str) + " | " + df['HABILIDADES'].astype(str) + " | " + df['ORIENTACOES_PEDAGOGICAS'].astype(str))

bncc_texts = concat_features_bncc(bncc_df_inf)
curriculo_texts = concat_features_curriculo(curriculo_df_inf)

# Carregar modelo de embeddings (LOCAL/OFFLINE)
try:
    print(f"🤖 Carregando modelo local...")
    if CACHE_FOLDER:
        model = SentenceTransformer(CAMINHO_FINAL_MODELO, cache_folder=CACHE_FOLDER)
    else:
        model = SentenceTransformer(CAMINHO_FINAL_MODELO)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    raise

# Gerar embeddings
try:
    print("🔄 Gerando embeddings da BNCC...")
    bncc_embeddings = model.encode(bncc_texts.tolist(), show_progress_bar=True)
    print("🔄 Gerando embeddings do currículo...")
    curriculo_embeddings = model.encode(curriculo_texts.tolist(), show_progress_bar=True)
    print("✅ Embeddings gerados com sucesso!")
except Exception as e:
    print(f"❌ Erro ao gerar embeddings: {e}")
    raise

# Calcular similaridades
print("🔄 Calculando similaridades...")
grau_similaridade = cosine_similarity(bncc_embeddings, curriculo_embeddings)
print("✅ Similaridades calculadas!")

# ==================================================================================
#                           ANÁLISE E RELATÓRIOS
# ==================================================================================

def encontrar_similaridade_balanceada(grau_similaridade, bncc_df, curriculo_df, nota_corte_inicial):
    """Encontra similaridades balanceadas por disciplina, evitando duplicatas"""
    relatorio_completo = []
    habilidades_ja_usadas = set()
    
    # Agrupar currículo por disciplina
    disciplinas_curriculo = {}
    for idx, linha in enumerate(curriculo_df.itertuples(index=False)):
        disciplina = getattr(linha, 'DISCIPLINA', getattr(linha, 'EIXO', 'SEM_DISCIPLINA'))
        if disciplina not in disciplinas_curriculo:
            disciplinas_curriculo[disciplina] = []
        
        if hasattr(linha, 'HABILIDADES'):
            codigo = extrair_codigo(linha.HABILIDADES)
        elif hasattr(linha, 'OBJETIVO_DE_APRENDIZAGEM'):
            codigo = extrair_codigo(linha.OBJETIVO_DE_APRENDIZAGEM)
        else:
            codigo = f"CURR_{idx}"
            
        disciplinas_curriculo[disciplina].append({'indice': idx, 'codigo': codigo, 'linha': linha})
    
    # Para cada habilidade BNCC
    for idx_bncc, linha_bncc in enumerate(bncc_df.itertuples(index=False)):
        similaridades_bncc = grau_similaridade[idx_bncc]
        
        if hasattr(linha_bncc, 'HABILIDADE'):
            bncc_codigo = extrair_codigo(linha_bncc.HABILIDADE)
            bncc_objetivo = linha_bncc.HABILIDADE
        else:
            bncc_codigo = f"BNCC_{idx_bncc}"
            bncc_objetivo = "OBJETIVO NÃO ENCONTRADO"
        
        correspondencias_por_disciplina = {}
        
        for disciplina, habilidades_disc in disciplinas_curriculo.items():
            melhores_da_disciplina = []
            for hab_curriculo in habilidades_disc:
                idx_curr = hab_curriculo['indice']
                codigo_curr = hab_curriculo['codigo']
                if codigo_curr in habilidades_ja_usadas: continue
                
                similaridade = similaridades_bncc[idx_curr]
                melhores_da_disciplina.append({
                    'indice': idx_curr,
                    'codigo': codigo_curr,
                    'similaridade': similaridade,
                    'linha': hab_curriculo['linha'],
                    'disciplina': disciplina
                })
            melhores_da_disciplina.sort(key=lambda x: x['similaridade'], reverse=True)
            correspondencias_por_disciplina[disciplina] = melhores_da_disciplina
        
        habilidades_similares = []
        nota_corte_usada = nota_corte_inicial
        
        # Estratégia 1: Melhor de cada disciplina
        for disciplina, candidatos in correspondencias_por_disciplina.items():
            if candidatos and candidatos[0]['similaridade'] >= nota_corte_inicial:
                melhor = candidatos[0]
                habilidades_similares.append(melhor)
                habilidades_ja_usadas.add(melhor['codigo'])
        
        # Estratégia 2: Busca adaptativa
        if not habilidades_similares:
            nota_corte_atual = nota_corte_inicial
            while not habilidades_similares and nota_corte_atual > 0.1:
                nota_corte_atual -= 0.01
                for disciplina, candidatos in correspondencias_por_disciplina.items():
                    for candidato in candidatos:
                        if (candidato['similaridade'] >= nota_corte_atual and 
                            candidato['codigo'] not in habilidades_ja_usadas):
                            habilidades_similares.append(candidato)
                            habilidades_ja_usadas.add(candidato['codigo'])
                            break
                if habilidades_similares:
                    nota_corte_usada = nota_corte_atual
                    break
        
        # Estratégia 3: Melhor geral
        if not habilidades_similares:
            todas_opcoes = []
            for disciplina, candidatos in correspondencias_por_disciplina.items():
                for candidato in candidatos:
                    if candidato['codigo'] not in habilidades_ja_usadas:
                        todas_opcoes.append(candidato)
            if todas_opcoes:
                melhor_geral = max(todas_opcoes, key=lambda x: x['similaridade'])
                habilidades_similares.append(melhor_geral)
                habilidades_ja_usadas.add(melhor_geral['codigo'])
                nota_corte_usada = melhor_geral['similaridade']
        
        # Estratégia 4: Adicionar até 3 matches
        if len(habilidades_similares) < 3:
            for disciplina, candidatos in correspondencias_por_disciplina.items():
                if len(habilidades_similares) >= 3: break
                for candidato in candidatos[1:]:
                    if (candidato['codigo'] not in habilidades_ja_usadas and 
                        candidato['similaridade'] >= nota_corte_usada * 0.9):
                        habilidades_similares.append(candidato)
                        habilidades_ja_usadas.add(candidato['codigo'])
                        break
        
        habilidades_similares.sort(key=lambda x: x['similaridade'], reverse=True)
        
        habilidade_bncc = {
            'bncc_indice': idx_bncc + 1,
            'bncc_codigo': bncc_codigo,
            'bncc_eixo': linha_bncc.EIXO,
            'bncc_objetivo': bncc_objetivo,
            'bncc_exemplos': getattr(linha_bncc, 'EXEMPLOS', 'N/A'),
            'habilidades_similares': [],
            'tem_similaridade_original': any(h['similaridade'] >= nota_corte_inicial for h in habilidades_similares),
            'nota_corte_usada': nota_corte_usada,
            'quantidade_similares': len(habilidades_similares),
            'maior_similaridade': max([h['similaridade'] for h in habilidades_similares]) if habilidades_similares else 0,
            'disciplinas_envolvidas': len(set(h['disciplina'] for h in habilidades_similares))
        }
        
        for similar in habilidades_similares:
            linha_curriculo = similar['linha']
            if hasattr(linha_curriculo, 'HABILIDADES'):
                curriculo_objetivo = linha_curriculo.HABILIDADES
                curriculo_exemplos = getattr(linha_curriculo, 'ORIENTACOES_PEDAGOGICAS', 'N/A')
            elif hasattr(linha_curriculo, 'OBJETIVO_DE_APRENDIZAGEM'):
                curriculo_objetivo = linha_curriculo.OBJETIVO_DE_APRENDIZAGEM
                curriculo_exemplos = getattr(linha_curriculo, 'EXEMPLOS', 'N/A')
            else:
                curriculo_objetivo = "OBJETIVO NÃO ENCONTRADO"
                curriculo_exemplos = "N/A"
            
            habilidade_similar = {
                'curriculo_indice': similar['indice'] + 1,
                'curriculo_codigo': similar['codigo'],
                'curriculo_eixo': similar['disciplina'],
                'curriculo_objetivo': curriculo_objetivo,
                'curriculo_exemplos': curriculo_exemplos,
                'similaridade': similar['similaridade']
            }
            habilidade_bncc['habilidades_similares'].append(habilidade_similar)
        
        relatorio_completo.append(habilidade_bncc)
        if (idx_bncc + 1) % 10 == 0:
            print(f"📈 Processadas {idx_bncc + 1}/{len(bncc_df)} habilidades BNCC")
            
    return relatorio_completo

NOTA_CORTE = CONFIGURACOES['NOTA_CORTE']
data_relatorio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"📊 Gerando relatório com nota de corte inicial de {NOTA_CORTE*100}%...")
print("🔄 Usando algoritmo balanceado (Anos Iniciais)...")

relatorio_completo = encontrar_similaridade_balanceada(grau_similaridade, bncc_df_inf, curriculo_df_inf, NOTA_CORTE)

# Estatísticas
estatisticas = {
    'total_bncc': len(bncc_df_inf),
    'bncc_com_similaridade_original': 0,
    'bncc_com_similaridade_adaptativa': 0,
    'total_matches_acima_corte': 0,
    'notas_corte_usadas': [],
    'nota_corte_original': NOTA_CORTE,
    'modelo_usado': CONFIGURACOES['MODELO_EMBEDDINGS'],
    'data_analise': data_relatorio
}

for hab in relatorio_completo:
    if hab['tem_similaridade_original']: estatisticas['bncc_com_similaridade_original'] += 1
    estatisticas['bncc_com_similaridade_adaptativa'] += 1
    estatisticas['notas_corte_usadas'].append(hab['nota_corte_usada'])
    for sim in hab['habilidades_similares']:
        if sim['similaridade'] >= NOTA_CORTE: estatisticas['total_matches_acima_corte'] += 1

print("✅ Relatório analisado! Salvando arquivos...")

# ==================================================================================
#                           SALVAMENTO
# ==================================================================================

def gerar_relatorio_texto():
    # Calcular estatísticas adicionais para o relatório
    codigos_unicos_usados = set()
    disciplinas_usadas = {}
    
    for hab_bncc in relatorio_completo:
        for similar in hab_bncc['habilidades_similares']:
            codigos_unicos_usados.add(similar['curriculo_codigo'])
            disc = similar['curriculo_eixo']
            disciplinas_usadas[disc] = disciplinas_usadas.get(disc, 0) + 1
            
    relatorio_txt = f"""RELATÓRIO DE SIMILARIDADE - ANOS INICIAIS
Data: {estatisticas['data_analise']}
Nota de corte: {estatisticas['nota_corte_original']*100}%
Modelo: {os.path.basename(str(estatisticas['modelo_usado']))}

ESTATÍSTICAS:
• Total BNCC: {estatisticas['total_bncc']}
• Matches Fortes (>= {estatisticas['nota_corte_original']*100}%): {estatisticas['bncc_com_similaridade_original']}
• Habilidades Currículo Usadas: {len(codigos_unicos_usados)}

DISTRIBUIÇÃO POR DISCIPLINA:
"""
    for disc, count in sorted(disciplinas_usadas.items()):
        relatorio_txt += f"• {disc}: {count}\n"

    relatorio_txt += "\nDETALHES:\n"
    for hab in relatorio_completo:
        relatorio_txt += f"\n{'='*80}\nBNCC: {hab['bncc_codigo']} - {hab['bncc_eixo']}\n"
        relatorio_txt += f"Obj: {hab['bncc_objetivo']}\n"
        relatorio_txt += f"Corte usado: {hab['nota_corte_usada']:.1%}\n"
        for sim in hab['habilidades_similares']:
            relatorio_txt += f"   -> {sim['curriculo_codigo']} ({sim['curriculo_eixo']}) | Sim: {sim['similaridade']:.1%}\n"
            relatorio_txt += f"      {sim['curriculo_objetivo']}\n"
            
    return relatorio_txt

def gerar_relatorio_csv():
    dados = []
    for hab in relatorio_completo:
        for sim in hab['habilidades_similares']:
            dados.append({
                'BNCC_Codigo': hab['bncc_codigo'],
                'BNCC_Eixo': hab['bncc_eixo'],
                'BNCC_Objetivo': hab['bncc_objetivo'],
                'Curriculo_Codigo': sim['curriculo_codigo'],
                'Curriculo_Eixo': sim['curriculo_eixo'],
                'Curriculo_Objetivo': sim['curriculo_objetivo'],
                'Similaridade': sim['similaridade'],
                'Similaridade_Percentual': f"{sim['similaridade']:.1%}",
                'Nota_Corte_Usada': f"{hab['nota_corte_usada']:.1%}"
            })
    return pd.DataFrame(dados)

# Criar pasta docs/anos iniciais
docs_path = os.path.join(DIRETORIO_ATUAL, "docs", "anos iniciais")
if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    print(f"📁 Pasta '{docs_path}' criada.")

prefixo = CONFIGURACOES['PREFIXO_ARQUIVOS']

try:
    with open(os.path.join(docs_path, f"anosiniciais_{prefixo}relatorio_completo.txt"), "w", encoding="utf-8") as f:
        f.write(gerar_relatorio_texto())
    
    df_csv = gerar_relatorio_csv()
    df_csv.to_csv(os.path.join(docs_path, f"anosiniciais_{prefixo}relatorio.csv"), index=False, encoding="utf-8-sig")
    
    print("✅ Relatórios salvos com sucesso!")
except Exception as e:
    print(f"❌ Erro ao salvar relatórios: {e}")

# Heatmap
if CONFIGURACOES['GERAR_HEATMAP']:
    try:
        bncc_cods = bncc_df_inf['HABILIDADE'].apply(extrair_codigo).tolist()
        curr_cods = curriculo_df_inf['HABILIDADES'].apply(extrair_codigo).tolist()
        sim_df = pd.DataFrame(grau_similaridade, index=bncc_cods, columns=curr_cods)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(sim_df.iloc[:25, :25], annot=True, fmt=".1f", cmap="Blues")
        plt.title(f"Heatmap (Top 25) - Anos Iniciais")
        plt.savefig(os.path.join(docs_path, f"anosiniciais_{prefixo}heatmap.png"), bbox_inches='tight')
        print("✅ Heatmap salvo.")
    except Exception as e:
        print(f"⚠️ Erro no heatmap: {e}")

print("="*80)
print("✅ PROCESSO FINALIZADO - ANOS INICIAIS")
print("="*80)