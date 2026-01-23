import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg') # Importante para não travar o servidor Flask
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================================================================================
#                           CONFIGURAÇÃO DE AMBIENTE E CAMINHOS
# ==================================================================================

# 1. Limpeza de Proxy (Garante modo offline)
for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(env_var, None)

# 2. Configuração de Diretórios (AJUSTADO PARA A PASTA CORE)
# Pega o diretório onde este arquivo (similarity.py) está: .../bncc_computadional/core
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
# Sobe um nível para achar a raiz do projeto: .../bncc_computadional
PROJECT_ROOT = os.path.dirname(CORE_DIR)

# Define onde estão os recursos baseados na Raiz
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

# Cria pasta docs se não existir
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# ==================================================================================
#                           FUNÇÕES DE UTILIDADE (MODELO E TEXTO)
# ==================================================================================

def encontrar_modelo_real(pasta_base):
    """Encontra a subpasta exata onde o modelo HuggingFace salvou os arquivos"""
    if not os.path.exists(pasta_base):
        return 'all-MiniLM-L6-v2'

    for root, dirs, files in os.walk(pasta_base):
        if 'config.json' in files and ('pytorch_model.bin' in files or 'model.safetensors' in files):
            return root
    return 'all-MiniLM-L6-v2'

def carregar_modelo():
    """Carrega o modelo garantindo uso offline se disponível"""
    try:
        caminho_modelo = encontrar_modelo_real(MODELS_DIR)
        print(f"🤖 Carregando modelo de: {caminho_modelo}")
        
        if os.path.isabs(caminho_modelo) or os.path.exists(caminho_modelo):
            return SentenceTransformer(caminho_modelo)
        else:
            return SentenceTransformer(caminho_modelo, cache_folder=MODELS_DIR)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return SentenceTransformer('all-MiniLM-L6-v2')

def extrair_codigo(text):
    """Extrai códigos como (EF01LP01) ou (EI03TS01)"""
    if pd.isna(text): return "(SEM_COD)"
    text = str(text)
    match = re.search(r'\(([A-Z0-9]{3,12})\)', text)
    if match: return f"({match.group(1)})"
    match_gen = re.search(r'^\(([^)]+)\)', text)
    if match_gen: return f"({match_gen.group(1)})"
    return f"({text[:10]}...)"

def detectar_colunas(df):
    """Detecta dinamicamente as colunas de Eixo, Habilidade e Exemplos"""
    cols = [c.strip().upper() for c in df.columns]
    
    eixo_possiveis = ['EIXO', 'DISCIPLINA', 'COMPONENTE', 'CAMPO DE EXPERIENCIA', 'CAMPO']
    hab_possiveis = ['HABILIDADE', 'HABILIDADES', 'OBJETIVO DE APRENDIZAGEM', 'OBJETIVOS', 'DIREITOS']
    ex_possiveis = ['EXEMPLOS', 'ORIENTACOES_PEDAGOGICAS', 'ORIENTAÇÕES_PEDAGÓGICAS', 'SABERES']
    
    eixo = next((df.columns[cols.index(c)] for c in eixo_possiveis if c in cols), None)
    habilidade = next((df.columns[cols.index(c)] for c in hab_possiveis if c in cols), None)
    exemplos = next((df.columns[cols.index(c)] for c in ex_possiveis if c in cols), None)
    
    if not eixo or not habilidade:
        # Fallback para arquivos simples
        if len(df.columns) >= 2:
            return df.columns[0], df.columns[1], df.columns[2] if len(df.columns) > 2 else 'EXEMPLOS_TEMP'
        raise ValueError(f"Colunas obrigatórias não encontradas. Disponíveis: {df.columns.tolist()}")
    
    if not exemplos:
        df['EXEMPLOS_TEMP'] = 'N/A'
        exemplos = 'EXEMPLOS_TEMP'
        
    return eixo, habilidade, exemplos

def processar_texto(df, e, h, x):
    """Prepara o texto para embedding"""
    return (df[e].fillna('GERAL').astype(str) + " | " + 
            df[h].fillna('N/A').astype(str) + " | " + 
            df[x].fillna('N/A').astype(str))

# ==================================================================================
#                           ALGORITMO BALANCEADO (O SEGREDO)
# ==================================================================================

def encontrar_similaridades_balanceada(sim_matriz, bncc_df, curr_df, nota_corte):
    """
    Algoritmo Inteligente dos scripts individuais:
    1. Tenta match > nota_corte.
    2. Se não achar, usa busca adaptativa (diminui a nota).
    3. Se ainda não achar, pega o melhor absoluto.
    Isso garante que NUNCA retorne vazio.
    """
    relatorio = []
    # Usado para tentar variar, mas permitimos reuso se for necessário para match forte
    codigos_usados = set() 
    
    # 1. Mapear Currículo (Agrupar por Disciplina/Eixo)
    eixo_curr_col, hab_curr_col, ex_curr_col = detectar_colunas(curr_df)
    
    disciplinas = {}
    for i, row in curr_df.iterrows():
        eixo = row.get(eixo_curr_col, 'GERAL')
        if eixo not in disciplinas: disciplinas[eixo] = []
        
        hab_txt = row.get(hab_curr_col, '')
        codigo = extrair_codigo(hab_txt)
        disciplinas[eixo].append({'i': i, 'codigo': codigo, 'row': row})
    
    # 2. Processar BNCC
    eixo_bncc_col, hab_bncc_col, ex_bncc_col = detectar_colunas(bncc_df)
    
    for i_bncc, row_bncc in bncc_df.iterrows():
        matches = []
        nota_corte_usada = nota_corte
        
        # --- ESTRATÉGIA 1: Match Perfeito por Disciplina ---
        for eixo_nome, candidatos in disciplinas.items():
            melhor = max(candidatos, key=lambda c: sim_matriz[i_bncc][c['i']])
            score = sim_matriz[i_bncc][melhor['i']]
            
            if score >= nota_corte:
                matches.append({
                    'curriculo_indice': melhor['i'] + 1,
                    'curriculo_codigo': melhor['codigo'],
                    'curriculo_eixo': eixo_nome,
                    'curriculo_objetivo': melhor['row'].get(hab_curr_col, ''),
                    'curriculo_exemplos': melhor['row'].get(ex_curr_col, ''),
                    'similaridade': score,
                    'tipo_match': 'Forte'
                })
                codigos_usados.add(melhor['codigo'])

        # --- ESTRATÉGIA 2 e 3: Fallback Adaptativo ---
        # Se a estratégia 1 não gerou nada, precisamos garantir pelo menos UM match
        if not matches:
            # Pega TODOS os candidatos de todas as disciplinas
            todos_candidatos = []
            for eixo_nome, lista in disciplinas.items():
                for c in lista:
                    todos_candidatos.append({'c': c, 'eixo': eixo_nome, 'score': sim_matriz[i_bncc][c['i']]})
            
            if todos_candidatos:
                # Pega o melhor absoluto, independente da nota de corte
                melhor_geral = max(todos_candidatos, key=lambda x: x['score'])
                matches.append({
                    'curriculo_indice': melhor_geral['c']['i'] + 1,
                    'curriculo_codigo': melhor_geral['c']['codigo'],
                    'curriculo_eixo': melhor_geral['eixo'],
                    'curriculo_objetivo': melhor_geral['c']['row'].get(hab_curr_col, ''),
                    'curriculo_exemplos': melhor_geral['c']['row'].get(ex_curr_col, ''),
                    'similaridade': melhor_geral['score'],
                    'tipo_match': 'Adaptativo'
                })
                nota_corte_usada = melhor_geral['score'] # Registra que tivemos que baixar a régua

        # Ordena matches pela maior similaridade
        matches.sort(key=lambda x: x['similaridade'], reverse=True)
        
        relatorio.append({
            'bncc_indice': i_bncc + 1,
            'bncc_codigo': extrair_codigo(row_bncc.get(hab_bncc_col)),
            'bncc_eixo': row_bncc.get(eixo_bncc_col),
            'bncc_objetivo': str(row_bncc.get(hab_bncc_col)),
            'bncc_exemplos': str(row_bncc.get(ex_bncc_col, '')),
            'nota_corte_efetiva': nota_corte_usada,
            'habilidades_similares': matches
        })
    
    return relatorio

# ==================================================================================
#                           FUNÇÃO PRINCIPAL (APP.PY)
# ==================================================================================

def process_uploaded_file(uploaded_path, segment, nota_corte):
    """
    Função principal chamada pelo Flask.
    """
    print(f"🚀 Iniciando Análise: {segment} | Corte Solicitado: {nota_corte:.0%}")
    
    # 1. Identificar arquivo BNCC (Na Raiz do Projeto)
    bncc_map = {
        'infantil': 'bncc_df_inf.xlsx',
        'educacao_infantil': 'bncc_df_inf.xlsx',
        'anos_iniciais': 'bncc_df_anosiniciais.xlsx',
        'anos_finais': 'bncc_df_anosfinais.xlsx',
        'ensinomedio': 'bncc_df_medio.xlsx'
    }
    
    key_segment = segment.lower().replace(" ", "_").strip()
    if 'iniciais' in key_segment: key_segment = 'anos_iniciais'
    elif 'finais' in key_segment: key_segment = 'anos_finais'
    elif 'infantil' in key_segment: key_segment = 'infantil'
    
    bncc_filename = bncc_map.get(key_segment, 'bncc_df_anosfinais.xlsx')
    bncc_path = os.path.join(PROJECT_ROOT, bncc_filename)
    
    print(f"📂 Arquivo BNCC: {bncc_path}")
    
    if not os.path.exists(bncc_path):
        raise FileNotFoundError(f"Arquivo BNCC não encontrado: {bncc_path}")

    # 2. Carregar DataFrames
    try:
        curr_df = pd.read_excel(uploaded_path)
        bncc_df = pd.read_excel(bncc_path)
        curr_df.columns = curr_df.columns.str.strip()
        bncc_df.columns = bncc_df.columns.str.strip()
    except Exception as e:
        raise Exception(f"Erro ao ler Excel: {str(e)}")

    # 3. Detectar Colunas e Texto
    print("🔍 Detectando colunas...")
    e_b, h_b, x_b = detectar_colunas(bncc_df)
    e_c, h_c, x_c = detectar_colunas(curr_df)
    
    bncc_txt = processar_texto(bncc_df, e_b, h_b, x_b)
    curr_txt = processar_texto(curr_df, e_c, h_c, x_c)

    # 4. IA e Embeddings
    model = carregar_modelo()
    print("🧠 Gerando Embeddings...")
    bncc_emb = model.encode(bncc_txt.tolist(), show_progress_bar=True)
    curr_emb = model.encode(curr_txt.tolist(), show_progress_bar=True)

    # 5. Similaridade
    print("🧮 Calculando Matriz...")
    sim_matriz = cosine_similarity(bncc_emb, curr_emb)

    # 6. Algoritmo Balanceado (AQUI ESTÁ A CORREÇÃO PRINCIPAL)
    print("🎯 Cruzando dados com algoritmo balanceado...")
    resultado_analise = encontrar_similaridades_balanceada(sim_matriz, bncc_df, curr_df, nota_corte)

    # 7. Gerar Estatísticas e Arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_segment = segment.replace(" ", "_").lower()
    filename_base = f"{safe_segment}_{timestamp}"
    
    # Nomes dos arquivos
    csv_filename = f"{filename_base}_relatorio.csv"
    txt_resumo_filename = f"{filename_base}_resumo_executivo.txt"
    txt_detalhado_filename = f"{filename_base}_relatorio_detalhado.txt" # O arquivo que faltava
    heatmap_filename = f"{filename_base}_heatmap.png"
    
    # Caminhos absolutos
    abs_csv = os.path.join(DOCS_DIR, csv_filename)
    abs_txt_resumo = os.path.join(DOCS_DIR, txt_resumo_filename)
    abs_txt_detalhado = os.path.join(DOCS_DIR, txt_detalhado_filename)
    abs_heatmap = os.path.join(DOCS_DIR, heatmap_filename)

    # Processar dados para CSV e Stats
    csv_rows = []
    habilidades_curr_usadas = set()
    disciplinas_map = set()
    notas_adaptativas = []
    matches_fortes_count = 0
    
    txt_detalhado_content = f"RELATÓRIO DETALHADO - {segment.upper()}\nData: {datetime.now()}\n\n"

    for item in resultado_analise:
        notas_adaptativas.append(item['nota_corte_efetiva'])
        
        txt_detalhado_content += f"{'='*80}\nBNCC: {item['bncc_codigo']} ({item['bncc_eixo']})\n"
        txt_detalhado_content += f"Objetivo: {item['bncc_objetivo'][:200]}...\n"
        txt_detalhado_content += f"Matches encontrados: {len(item['habilidades_similares'])}\n"
        
        for match in item['habilidades_similares']:
            habilidades_curr_usadas.add(match['curriculo_codigo'])
            disciplinas_map.add(match['curriculo_eixo'])
            
            if match['similaridade'] >= nota_corte:
                matches_fortes_count += 1
                
            csv_rows.append({
                'BNCC_Codigo': item['bncc_codigo'],
                'BNCC_Objetivo': item['bncc_objetivo'],
                'BNCC_Eixo': item['bncc_eixo'],
                'Curriculo_Codigo': match['curriculo_codigo'],
                'Curriculo_Objetivo': match['curriculo_objetivo'],
                'Curriculo_Eixo': match['curriculo_eixo'],
                'Curriculo_Exemplos': match.get('curriculo_exemplos', ''),
                'Similaridade': f"{match['similaridade']:.1%}",
                'Valor_Sim': match['similaridade'],
                'Tipo_Match': match.get('tipo_match', 'N/A')
            })
            
            txt_detalhado_content += f"   -> {match['curriculo_codigo']} | {match['curriculo_eixo']} | Sim: {match['similaridade']:.1%}\n"
            txt_detalhado_content += f"      {match['curriculo_objetivo'][:150]}...\n"
        txt_detalhado_content += "\n"

    # Salvar CSV
    pd.DataFrame(csv_rows).to_csv(abs_csv, index=False, encoding='utf-8-sig')
    
    # Salvar Detalhado (O que estava faltando)
    with open(abs_txt_detalhado, 'w', encoding='utf-8') as f:
        f.write(txt_detalhado_content)

    # Salvar Resumo Executivo
    media_adaptativa = np.mean(notas_adaptativas) if notas_adaptativas else 0
    with open(abs_txt_resumo, 'w', encoding='utf-8') as f:
        f.write(f"RESUMO EXECUTIVO - {segment.upper()}\n")
        f.write(f"Nota de Corte Solicitada: {nota_corte:.0%}\n")
        f.write(f"Nota Média Efetiva (Adaptativa): {media_adaptativa:.1%}\n\n")
        f.write(f"Total BNCC: {len(bncc_df)}\n")
        f.write(f"Total Currículo: {len(curr_df)}\n")
        f.write(f"Habilidades Currículo Utilizadas: {len(habilidades_curr_usadas)}\n")
        f.write(f"Disciplinas Mapeadas: {len(disciplinas_map)}\n")

    # Gerar Heatmap
    try:
        plt.figure(figsize=(14, 10))
        limite = min(25, len(bncc_df))
        sns.heatmap(sim_matriz[:limite, :limite], cmap='Blues', annot=True, fmt='.1f')
        plt.title(f"Heatmap (Top {limite})")
        plt.tight_layout()
        plt.savefig(abs_heatmap)
        plt.close()
    except Exception as e:
        print(f"⚠️ Erro heatmap: {e}")

    print("✅ Processamento concluído!")

    # Retorno para o Flask (chaves que o app.py usa)
    return {
        'resumo': {
            'total_bncc': len(bncc_df),
            'total_curriculo': len(curr_df),
            'habilidades_utilizadas': len(habilidades_curr_usadas), # Corrigido para mostrar usadas
            'disciplinas_mapeadas': len(disciplinas_map), # Adicionado
            'media_adaptativa': f"{media_adaptativa:.1%}", # Adicionado
            'segmento': segment
        },
        'files': {
            'csv': f"docs/{csv_filename}",
            'resumo_executivo': f"docs/{txt_resumo_filename}",
            'relatorio_detalhado': f"docs/{txt_detalhado_filename}", # Adicionado
            'heatmap': f"docs/{heatmap_filename}"
        },
        'top_matches': sorted(csv_rows, key=lambda x: x['Valor_Sim'], reverse=True)[:10]
    }

if __name__ == "__main__":
    print("Este arquivo é um módulo core. Execute app.py.")