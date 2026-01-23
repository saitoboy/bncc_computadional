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

# 2. Define os arquivos Excel (INFANTIL)
CAMINHO_BNCC = os.path.join(DIRETORIO_ATUAL, "bncc_df_inf.xlsx")
CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "CURRICULO_ESTADO_educacao_infantil.xlsx")

# 3. Função para CAÇAR o modelo dentro da estrutura de pastas
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
print("🚀 ANALISADOR DE SIMILARIDADE - EDUCAÇÃO INFANTIL (OFFLINE/LOCAL)")
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
    print(f"✅ Dados carregados: {len(bncc_df_inf)} objetivos BNCC, {len(curriculo_df_inf)} objetivos currículo")
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    raise

# Detecção inteligente de colunas (Adaptado para Infantil: Campos de Experiência / Objetivos)
def detectar_colunas(df):
    cols = [c.strip().upper() for c in df.columns]
    
    # Possíveis nomes para Campos de Experiência (Eixo)
    eixo_possiveis = ['EIXO', 'CAMPO DE EXPERIENCIA', 'CAMPOS DE EXPERIENCIA', 'CAMPO', 'DISCIPLINA']
    # Possíveis nomes para Objetivos (Habilidade)
    hab_possiveis = ['OBJETIVO DE APRENDIZAGEM', 'OBJETIVOS', 'HABILIDADE', 'HABILIDADES', 'DIREITOS']
    # Possíveis nomes para Exemplos/Saberes
    ex_possiveis = ['EXEMPLOS', 'SABERES', 'CONTEUDOS', 'ORIENTACOES', 'ORIENTAÇÕES']
    
    eixo = next((df.columns[cols.index(c)] for c in eixo_possiveis if c in cols), None)
    habilidade = next((df.columns[cols.index(c)] for c in hab_possiveis if c in cols), None)
    exemplos = next((df.columns[cols.index(c)] for c in ex_possiveis if c in cols), None)
    
    if not eixo or not habilidade:
        raise ValueError(f"Colunas obrigatórias (Campo/Objetivo) não encontradas. Disponíveis: {df.columns.tolist()}")
    
    if not exemplos:
        df['EXEMPLOS_TEMP'] = 'N/A'
        exemplos = 'EXEMPLOS_TEMP'
        
    return eixo, habilidade, exemplos

# Normaliza os nomes das colunas
curriculo_df_inf.columns = curriculo_df_inf.columns.str.strip()
bncc_df_inf.columns = bncc_df_inf.columns.str.strip()

print("🔍 Detectando colunas BNCC...")
eixo_bncc, hab_bncc, ex_bncc = detectar_colunas(bncc_df_inf)
print(f"   -> Eixo: {eixo_bncc} | Objetivo: {hab_bncc}")

print("🔍 Detectando colunas Currículo...")
eixo_curr, hab_curr, ex_curr = detectar_colunas(curriculo_df_inf)
print(f"   -> Eixo: {eixo_curr} | Objetivo: {hab_curr}")

# ==================================================================================
#                    FUNÇÕES AUXILIARES
# ==================================================================================

def extrair_codigo(obj):
    if pd.isna(obj): return "(SEM_COD)"
    obj_str = str(obj)
    # Padrão EI (EI02TS01, etc)
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

def concat_features(df, col_eixo, col_hab, col_ex):
    return (df[col_eixo].astype(str) + " | " + df[col_hab].astype(str) + " | " + df[col_ex].astype(str))

bncc_texts = concat_features(bncc_df_inf, eixo_bncc, hab_bncc, ex_bncc)
curriculo_texts = concat_features(curriculo_df_inf, eixo_curr, hab_curr, ex_curr)

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
    print("🔄 Gerando embeddings...")
    bncc_embeddings = model.encode(bncc_texts.tolist(), show_progress_bar=True)
    curriculo_embeddings = model.encode(curriculo_texts.tolist(), show_progress_bar=True)
except Exception as e:
    print(f"❌ Erro ao gerar embeddings: {e}")
    raise

# Calcular similaridades
print("🔄 Calculando similaridades...")
grau_similaridade = cosine_similarity(bncc_embeddings, curriculo_embeddings)

# ==================================================================================
#                           ALGORITMO BALANCEADO
# ==================================================================================

def encontrar_similaridade_balanceada(grau_similaridade, bncc_df, curriculo_df, nota_corte_inicial):
    """Encontra similaridades balanceadas por Campo de Experiência, evitando duplicatas"""
    relatorio_completo = []
    codigos_ja_usados = set()
    
    # Agrupar currículo por Campo de Experiência (Eixo)
    campos_curriculo = {}
    
    col_eixo_c, col_hab_c, _ = detectar_colunas(curriculo_df)
    
    for idx, linha in enumerate(curriculo_df.itertuples(index=False)):
        # Pega o valor da coluna detectada dinamicamente
        campo = getattr(linha, col_eixo_c, 'GERAL') 
        if campo not in campos_curriculo:
            campos_curriculo[campo] = []
        
        hab_texto = getattr(linha, col_hab_c, str(idx))
        codigo = extrair_codigo(hab_texto)
            
        campos_curriculo[campo].append({'indice': idx, 'codigo': codigo, 'linha': linha})
    
    col_eixo_b, col_hab_b, col_ex_b = detectar_colunas(bncc_df)

    # Para cada objetivo BNCC
    for idx_bncc, linha_bncc in enumerate(bncc_df.itertuples(index=False)):
        similaridades_bncc = grau_similaridade[idx_bncc]
        
        bncc_texto_hab = getattr(linha_bncc, col_hab_b, '')
        bncc_codigo = extrair_codigo(bncc_texto_hab)
        bncc_eixo = getattr(linha_bncc, col_eixo_b, 'N/A')
        
        correspondencias_por_campo = {}
        
        for campo, objetivos_curr in campos_curriculo.items():
            melhores_do_campo = []
            for obj_curr in objetivos_curr:
                idx_curr = obj_curr['indice']
                codigo_curr = obj_curr['codigo']
                if codigo_curr in codigos_ja_usados: continue
                
                sim = similaridades_bncc[idx_curr]
                melhores_do_campo.append({
                    'indice': idx_curr,
                    'codigo': codigo_curr,
                    'similaridade': sim,
                    'linha': obj_curr['linha'],
                    'campo': campo
                })
            melhores_do_campo.sort(key=lambda x: x['similaridade'], reverse=True)
            correspondencias_por_campo[campo] = melhores_do_campo
        
        habilidades_similares = []
        nota_corte_usada = nota_corte_inicial
        
        # Estratégia 1: Melhor de cada Campo
        for campo, candidatos in correspondencias_por_campo.items():
            if candidatos and candidatos[0]['similaridade'] >= nota_corte_inicial:
                melhor = candidatos[0]
                habilidades_similares.append(melhor)
                codigos_ja_usados.add(melhor['codigo'])
        
        # Estratégia 2: Busca adaptativa
        if not habilidades_similares:
            nota_corte_atual = nota_corte_inicial
            while not habilidades_similares and nota_corte_atual > 0.1:
                nota_corte_atual -= 0.01
                for campo, candidatos in correspondencias_por_campo.items():
                    for candidato in candidatos:
                        if (candidato['similaridade'] >= nota_corte_atual and 
                            candidato['codigo'] not in codigos_ja_usados):
                            habilidades_similares.append(candidato)
                            codigos_ja_usados.add(candidato['codigo'])
                            break
                if habilidades_similares:
                    nota_corte_usada = nota_corte_atual
                    break
        
        # Estratégia 3: Melhor geral (se ainda vazio)
        if not habilidades_similares:
            todas = []
            for cp, cands in correspondencias_por_campo.items():
                for c in cands:
                    if c['codigo'] not in codigos_ja_usados: all_options = todas.append(c)
            if todas:
                best = max(todas, key=lambda x: x['similaridade'])
                habilidades_similares.append(best)
                codigos_ja_usados.add(best['codigo'])
                nota_corte_usada = best['similaridade']

        habilidades_similares.sort(key=lambda x: x['similaridade'], reverse=True)
        
        bncc_exemplo_val = getattr(linha_bncc, col_ex_b, 'N/A')

        item_relatorio = {
            'bncc_indice': idx_bncc + 1,
            'bncc_codigo': bncc_codigo,
            'bncc_eixo': bncc_eixo,
            'bncc_objetivo': bncc_texto_hab,
            'bncc_exemplos': bncc_exemplo_val,
            'habilidades_similares': [],
            'tem_similaridade_original': any(h['similaridade'] >= nota_corte_inicial for h in habilidades_similares),
            'nota_corte_usada': nota_corte_usada,
            'quantidade_similares': len(habilidades_similares),
            'maior_similaridade': max([h['similaridade'] for h in habilidades_similares]) if habilidades_similares else 0
        }
        
        for sim in habilidades_similares:
            linha_c = sim['linha']
            curr_hab_txt = getattr(linha_c, col_hab_c, 'N/A')
            curr_ex_txt = getattr(linha_c, ex_curr, 'N/A') # ex_curr vem do escopo global detectado anteriormente
            
            item_relatorio['habilidades_similares'].append({
                'curriculo_indice': sim['indice'] + 1,
                'curriculo_codigo': sim['codigo'],
                'curriculo_eixo': sim['campo'],
                'curriculo_objetivo': curr_hab_txt,
                'curriculo_exemplos': curr_ex_txt,
                'similaridade': sim['similaridade']
            })
        
        relatorio_completo.append(item_relatorio)
            
    return relatorio_completo

NOTA_CORTE = CONFIGURACOES['NOTA_CORTE']
data_relatorio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"📊 Gerando relatório (Corte: {NOTA_CORTE*100}%)...")
print("🔄 Usando algoritmo balanceado para Educação Infantil...")

relatorio_completo = encontrar_similaridade_balanceada(grau_similaridade, bncc_df_inf, curriculo_df_inf, NOTA_CORTE)

# Estatísticas
estatisticas = {
    'total_bncc': len(bncc_df_inf),
    'bncc_com_similaridade_original': 0,
    'total_matches_acima_corte': 0,
    'notas_corte_usadas': [],
    'nota_corte_original': NOTA_CORTE,
    'modelo_usado': 'Local Model',
    'data_analise': data_relatorio
}

for hab in relatorio_completo:
    if hab['tem_similaridade_original']: estatisticas['bncc_com_similaridade_original'] += 1
    estatisticas['notas_corte_usadas'].append(hab['nota_corte_usada'])
    for sim in hab['habilidades_similares']:
        if sim['similaridade'] >= NOTA_CORTE: estatisticas['total_matches_acima_corte'] += 1

print("✅ Análise concluída! Salvando arquivos...")

# ==================================================================================
#                           SALVAMENTO
# ==================================================================================

# Criar pasta docs/infantil
docs_path = os.path.join(DIRETORIO_ATUAL, "docs", "infantil")
if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    print(f"📁 Pasta '{docs_path}' criada.")

prefixo = CONFIGURACOES['PREFIXO_ARQUIVOS']

def gerar_relatorio_texto():
    txt = f"""RELATÓRIO INFANTIL - BNCC X CURRÍCULO
Data: {estatisticas['data_analise']}
Nota de Corte: {estatisticas['nota_corte_original']:.0%}

RESUMO:
• Objetivos BNCC: {estatisticas['total_bncc']}
• Matches Fortes: {estatisticas['bncc_com_similaridade_original']}
• Total Conexões Fortes: {estatisticas['total_matches_acima_corte']}

DETALHAMENTO:
"""
    for hab in relatorio_completo:
        txt += f"\n{'='*60}\nBNCC: {hab['bncc_codigo']} | {hab['bncc_eixo']}\n"
        txt += f"Obj: {hab['bncc_objetivo']}\n"
        txt += f"Corte Usado: {hab['nota_corte_usada']:.1%}\n"
        for s in hab['habilidades_similares']:
            txt += f"   -> {s['curriculo_codigo']} ({s['curriculo_eixo']}) | Sim: {s['similaridade']:.1%}\n"
            txt += f"      {s['curriculo_objetivo']}\n"
    return txt

def gerar_csv():
    dados = []
    for h in relatorio_completo:
        for s in h['habilidades_similares']:
            dados.append({
                'BNCC_Cod': h['bncc_codigo'],
                'BNCC_Eixo': h['bncc_eixo'],
                'BNCC_Obj': h['bncc_objetivo'],
                'Curr_Cod': s['curriculo_codigo'],
                'Curr_Eixo': s['curriculo_eixo'],
                'Curr_Obj': s['curriculo_objetivo'],
                'Similaridade': f"{s['similaridade']:.1%}",
                'Valor_Sim': s['similaridade']
            })
    return pd.DataFrame(dados)

try:
    with open(os.path.join(docs_path, f"infantil_{prefixo}relatorio.txt"), "w", encoding="utf-8") as f:
        f.write(gerar_relatorio_texto())
        
    df = gerar_csv()
    df.to_csv(os.path.join(docs_path, f"infantil_{prefixo}dados.csv"), index=False, encoding="utf-8-sig")
    
    # Resumo Executivo Rápido
    with open(os.path.join(docs_path, f"infantil_{prefixo}resumo.txt"), "w", encoding="utf-8") as f:
        f.write(f"RESUMO EXECUTIVO INFANTIL\n\n")
        f.write(f"Total BNCC: {len(bncc_df_inf)}\n")
        f.write(f"Cobertura com Nota Original: {estatisticas['bncc_com_similaridade_original']}\n")
        f.write(f"Média de Corte Usada: {np.mean(estatisticas['notas_corte_usadas']):.1%}\n\n")
        f.write("TOP 10 MATCHES:\n")
        top = df.nlargest(10, 'Valor_Sim')
        for _, row in top.iterrows():
            f.write(f"{row['BNCC_Cod']} <-> {row['Curr_Cod']} : {row['Similaridade']}\n")
            
    print("✅ Relatórios salvos.")
except Exception as e:
    print(f"❌ Erro ao salvar: {e}")

# Heatmap
if CONFIGURACOES['GERAR_HEATMAP']:
    try:
        bncc_cods = bncc_df_inf[hab_bncc].apply(extrair_codigo).tolist()
        curr_cods = curriculo_df_inf[hab_curr].apply(extrair_codigo).tolist()
        sim_df = pd.DataFrame(grau_similaridade, index=bncc_cods, columns=curr_cods)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(sim_df.iloc[:25, :25], annot=True, fmt=".1f", cmap="Blues")
        plt.title("Heatmap Infantil (Top 25)")
        plt.savefig(os.path.join(docs_path, f"infantil_{prefixo}heatmap.png"), bbox_inches='tight')
        print("✅ Heatmap gerado.")
    except Exception as e:
        print(f"⚠️ Erro no heatmap: {e}")

print("="*80)
print("✅ PROCESSO FINALIZADO - EDUCAÇÃO INFANTIL")
print("="*80)