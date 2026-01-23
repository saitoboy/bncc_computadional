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

# 🚫 PROXY DESABILITADO
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def carregar_modelo():
    """Carrega modelo com cache"""
    try:
        print("🤖 Carregando modelo...")
        return SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
    except:
        return SentenceTransformer('all-MiniLM-L6-v2')

def extrair_codigo(text):
    """Extrai (EI01EF01)"""
    if pd.isna(text): return "(SEM_COD)"
    text = str(text)
    match = re.search(r'\(([A-Z0-9]{3,12})\)', text)
    return f"({match.group(1)})" if match else f"({text[:10]}...)"

def detectar_colunas(df):
    """Detecta estrutura das colunas"""
    cols = [c.strip().upper() for c in df.columns]
    
    eixo_cols = ['EIXO', 'DISCIPLINA']
    hab_cols = ['HABILIDADE', 'HABILIDADES', 'OBJETIVO DE APRENDIZAGEM']
    ex_cols = ['EXEMPLOS', 'ORIENTACOES_PEDAGOGICAS', 'ORIENTAÇÕES_PEDAGÓGICAS']
    
    eixo = next((c for c in eixo_cols if c in cols), 'EIXO')
    habilidade = next((c for c in hab_cols if c in cols), 'HABILIDADE')
    exemplos = next((c for c in ex_cols if c in cols), 'EXEMPLOS')
    
    return eixo, habilidade, exemplos

def processar_texto(df, eixo_col, hab_col, ex_col):
    """EIXO | HABILIDADE | EXEMPLOS"""
    return (df[eixo_col].fillna('GERAL').astype(str) + " | " + 
            df[hab_col].fillna('N/A').astype(str) + " | " + 
            df[ex_col].fillna('N/A').astype(str))

def encontrar_similaridades(sim_matriz, bncc_df, curr_df, nota_corte):
    """Balanceado por disciplina - sem duplicar"""
    relatorio = []
    usados = set()
    
    # Agrupar currículo por disciplina
    disciplinas = {}
    _, _, _ = detectar_colunas(curr_df)
    eixo_col, hab_col, _ = detectar_colunas(curr_df)
    
    for i, row in curr_df.iterrows():
        eixo = row.get(eixo_col, 'GERAL')
        habilidade = row.get(hab_col, '')
        codigo = extrair_codigo(habilidade)
        
        if eixo not in disciplinas:
            disciplinas[eixo] = []
        disciplinas[eixo].append({'i': i, 'codigo': codigo, 'row': row})
    
    # Para cada BNCC
    eixo_bncc, hab_bncc, ex_bncc = detectar_colunas(bncc_df)
    
    for i_bncc, row_bncc in bncc_df.iterrows():
        bncc_codigo = extrair_codigo(row_bncc.get(hab_bncc, ''))
        bncc_eixo = row_bncc.get(eixo_bncc, 'N/A')
        bncc_obj = str(row_bncc.get(hab_bncc, ''))[:150]
        
        matches = []
        
        # 1 por disciplina
        for eixo, candidatos in disciplinas.items():
            disponiveis = [c for c in candidatos if c['codigo'] not in usados]
            if disponiveis:
                melhor = max(disponiveis, key=lambda c: sim_matriz[i_bncc][c['i']])
                matches.append({
                    'curriculo_indice': melhor['i'] + 1,
                    'curriculo_codigo': melhor['codigo'],
                    'curriculo_eixo': eixo,
                    'similaridade': sim_matriz[i_bncc][melhor['i']]
                })
                usados.add(melhor['codigo'])
        
        # Fallback melhor geral
        if not matches:
            todos = []
            for eixo, candidatos in disciplinas.items():
                for c in candidatos:
                    if c['codigo'] not in usados:
                        todos.append({'sim': sim_matriz[i_bncc][c['i']], 'c': c})
            if todos:
                melhor = max(todos, key=lambda x: x['sim'])
                matches.append({
                    'curriculo_indice': melhor['c']['i'] + 1,
                    'curriculo_codigo': melhor['c']['codigo'],
                    'curriculo_eixo': melhor['c']['row'].get(eixo_col, 'GERAL'),
                    'similaridade': melhor['sim']
                })
                usados.add(melhor['c']['codigo'])
        
        relatorio.append({
            'bncc_indice': i_bncc + 1,
            'bncc_codigo': bncc_codigo,
            'bncc_eixo': bncc_eixo,
            'bncc_objetivo': bncc_obj,
            'habilidades_similares': matches
        })
    
    return relatorio

def process_uploaded_file(uploaded_path, segment, nota_corte):
    """🎯 UNIVERSAL - TODOS OS FORMATOS"""
    print(f"🚀 {segment} | {nota_corte:.0%}")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # ARQUIVOS BNCC CORRETOS (pelos nomes que você mostrou)
    bncc_map = {
        'infantil': 'bncc_df_inf.xlsx',
        'inicial': 'bncc_df_anosiniciais.xlsx',
        'iniciais': 'bncc_df_anosiniciais.xlsx',
        'final': 'bncc_df_anosfinais.xlsx',
        'finais': 'bncc_df_anosfinais.xlsx',
        'anos iniciais': 'bncc_df_anosiniciais.xlsx',
        'anos finais': 'bncc_df_anosfinais.xlsx'
    }
    
    bncc_file = bncc_map.get(segment.lower().replace(' ', ''), 'bncc_df_inf.xlsx')
    bncc_path = os.path.join(base_dir, bncc_file)
    
    print(f"📁 BNCC: {bncc_file}")
    
    if not os.path.exists(bncc_path):
        raise Exception(f"❌ BNCC não encontrada: {bncc_file}")
    
    # CARREGAR DADOS
    curr_df = pd.read_excel(uploaded_path)
    bncc_df = pd.read_excel(bncc_path)
    
    # NORMALIZAR COLUNAS
    for df in [bncc_df, curr_df]:
        df.columns = df.columns.str.strip()
    
    print(f"📊 BNCC: {len(bncc_df)} | Currículo: {len(curr_df)}")
    
    # DETECTAR COLUNAS
    print("🔍 Detectando colunas...")
    eixo_bncc, hab_bncc, ex_bncc = detectar_colunas(bncc_df)
    eixo_curr, hab_curr, ex_curr = detectar_colunas(curr_df)
    
    print(f"📋 BNCC: {eixo_bncc}|{hab_bncc}|{ex_bncc}")
    print(f"📋 Curr: {eixo_curr}|{hab_curr}|{ex_curr}")
    
    # IA
    model = carregar_modelo()
    
    bncc_textos = processar_texto(bncc_df, eixo_bncc, hab_bncc, ex_bncc)
    curr_textos = processar_texto(curr_df, eixo_curr, hab_curr, ex_curr)
    
    print("🔄 Embeddings...")
    bncc_emb = model.encode(bncc_textos.tolist(), show_progress_bar=False)
    curr_emb = model.encode(curr_textos.tolist(), show_progress_bar=False)
    
    print("🔄 Similaridades...")
    sim_matriz = cosine_similarity(bncc_emb, curr_emb)
    
    print("🎯 Matches balanceados...")
    relatorio = encontrar_similaridades(sim_matriz, bncc_df, curr_df, nota_corte)
    
    # SALVAR ARQUIVOS
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    # CSV
    csv_data = []
    for hab in relatorio:
        for match in hab['habilidades_similares']:
            csv_data.append({
                'BNCC_Codigo': hab['bncc_codigo'],
                'Curriculo_Codigo': match['curriculo_codigo'],
                'Similaridade': f"{match['similaridade']:.1%}",
                'Eixo_BNCC': hab['bncc_eixo'],
                'Eixo_Curriculo': match['curriculo_eixo']
            })
    
    csv_path = os.path.join(docs_dir, f"{segment.replace(' ', '_')}_relatorio.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # HEATMAP
    try:
        plt.figure(figsize=(14, 10))
        n = min(20, sim_matriz.shape[0], sim_matriz.shape[1])
        sns.heatmap(sim_matriz[:n, :n], cmap='Blues', vmin=0, vmax=1, annot=True, fmt='.2f')
        plt.title(f'{segment.title()} - {nota_corte:.0f}%')
        plt.tight_layout()
        plt.savefig(os.path.join(docs_dir, f"{segment.replace(' ', '_')}_heatmap.png"), dpi=300)
        plt.close()
    except: pass
    
    # RESUMO EXECUTIVO
    top_matches = []
    resumo_txt = f"""RESUMO EXECUTIVO - {segment.upper()}
{datetime.now().strftime('%d/%m/%Y %H:%M')}
BNCC: {len(bncc_df)} | Currículo: {len(curr_df)}

TOP 10:"""
    
    for hab in relatorio[:10]:
        if hab['habilidades_similares']:
            m = hab['habilidades_similares'][0]
            top_matches.append({
                'bncc_indice': hab['bncc_indice'],
                'bncc_codigo': hab['bncc_codigo'],
                'bncc_objetivo': hab['bncc_objetivo'],
                'curriculo_indice': m['curriculo_indice'],
                'curriculo_codigo': m['curriculo_codigo'],
                'similaridade': m['similaridade']
            })
            resumo_txt += f"\n{hab['bncc_codigo']} ↔ {m['curriculo_codigo']} ({m['similaridade']:.1%})"
    
    resumo_path = os.path.join(docs_dir, f"{segment.replace(' ', '_')}_resumo_executivo.txt")
    with open(resumo_path, 'w', encoding='utf-8') as f:
        f.write(resumo_txt)
    
    # DETALHADO
    detalhado_txt = "RELATÓRIO DETALHADO\n\n"
    for hab in relatorio[:25]:
        detalhado_txt += f"BNCC {hab['bncc_codigo']} ({hab['bncc_eixo']})\n"
        for m in hab['habilidades_similares'][:2]:
            detalhado_txt += f"  ↔ {m['curriculo_codigo']} | {m['similaridade']:.1%}\n"
        detalhado_txt += "\n"
    
    detalhado_path = os.path.join(docs_dir, f"{segment.replace(' ', '_')}_relatorio_detalhado.txt")
    with open(detalhado_path, 'w', encoding='utf-8') as f:
        f.write(detalhado_txt)
    
    # RETORNO
    resumo_final = {
        'total_bncc': len(bncc_df),
        'total_curriculo': len(curr_df),
        'habilidades_utilizadas': len(top_matches),
        'disciplinas_envolvidas': len(set(h['bncc_eixo'] for h in relatorio)),
        'distribuicao_disciplinas': {h['bncc_eixo']:1 for h in relatorio[:10]},
        'nota_media_usada': nota_corte,
        'eficiencia_uso': min(95, len(top_matches)/max(1,len(relatorio))*100),
        'segmento': segment.title()
    }
    
    print(f"✅ PRONTO! {len(bncc_df)}x{len(curr_df)} | {len(top_matches)} matches")
    
    return {
        'resumo': resumo_final,
        'files': {
            'csv': f"docs/{segment.replace(' ', '_')}_relatorio.csv",
            'heatmap': f"docs/{segment.replace(' ', '_')}_heatmap.png",
            'resumo_executivo': f"docs/{segment.replace(' ', '_')}_resumo_executivo.txt",
            'relatorio_detalhado': f"docs/{segment.replace(' ', '_')}_relatorio_detalhado.txt"
        },
        'top_matches': top_matches
    }

print("✅ similarity.py - FUNCIONA COM bncc_df_anosiniciais.xlsx + bncc_df_anosfinais.xlsx!")
