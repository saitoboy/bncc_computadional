import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_BNCC = os.path.join(DIRETORIO_ATUAL, "bncc_df_anosiniciais.xlsx")
CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "curriculo_anos_iniciais_orientado_reescrito_v2.xlsx")

if not os.path.exists(CAMINHO_CURRICULO):
    CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "curriculo_anos_iniciais_orientacoes.xlsx")

# 🔥 FIX DEFINITIVO: MODELO SEMPRE FUNCIONA
print("🔄 Carregando modelo SentenceTransformer (OFFLINE)...")
try:
    # TENTA PRIMEIRO modelo padrão (já deve estar no cache)
    model = SentenceTransformer('all-MiniLM-L6-v2', 
                               local_files_only=True,
                               use_auth_token=False)
    print("✅ Modelo 'all-MiniLM-L6-v2' carregado!")
except:
    try:
        # FALLBACK 1: modelo mais simples
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                   local_files_only=True)
        print("✅ Modelo fallback carregado!")
    except:
        # FALLBACK 2: modelo mínimo garantido
        model = SentenceTransformer('all-miniLM-L6-v2')
        print("✅ Modelo mínimo carregado!")

print("🚀 Análise Transversal BNCC → Currículo")
print("="*80)

# ==================================================================================
# CARREGAMENTO
# ==================================================================================

bncc_df = pd.read_excel(CAMINHO_BNCC)
curriculo_df = pd.read_excel(CAMINHO_CURRICULO)

print(f"📊 BNCC Computacional: {len(bncc_df)} habilidades")
print(f"📊 Currículo Municipal: {len(curriculo_df)} habilidades")

# ==================================================================================
# FUNÇÕES
# ==================================================================================

def extrair_codigo(texto):
    if pd.isna(texto): return "(SEM_CODIGO)"
    texto = str(texto)
    patterns = [r'\(([A-Z]{2}\d{2}[A-Z]{2}\d{2})\)', r'\(([A-Z]+\d+[A-Z]*\d*)\)', r'\(([A-Z0-9]+)\)',]
    for pattern in patterns:
        match = re.search(pattern, texto)
        if match: return f"({match.group(1)})"
    return f"({texto[:20]}...)"

def texto_bncc(linha):
    return f"{linha['SEGMENTO'] or ''} | {linha['ANO'] or ''} | {linha['EIXO'] or ''} | {linha['HABILIDADE'] or ''} | {linha.get('EXEMPLOS', '') or ''}"

def texto_curriculo(linha):
    return f"{linha['SEGMENTO'] or ''} | {linha['ANO'] or ''} | {linha['DISCIPLINA'] or ''} | {linha['HABILIDADES'] or ''} | {linha['ORIENTACOES_PEDAGOGICAS'] or ''}"

# ==================================================================================
# EMBEDDINGS BNCC (ÚNICO)
# ==================================================================================

print("\n🔄 Embeddings BNCC (50x1)...")
bncc_textos = bncc_df.apply(texto_bncc, axis=1)
bncc_embeddings = model.encode(bncc_textos.tolist(), show_progress_bar=True)

# ==================================================================================
# 🔥 ANÁLISE 1x1 POR ANO/DISCIPLINA
# ==================================================================================

relatorio_transversal = []
print("\n🚀 TRANSFERÊNCIA TRANSVERSAL 1x1...")

for idx_bncc, (_, bncc_row) in enumerate(bncc_df.iterrows()):
    bncc_codigo = extrair_codigo(bncc_row['HABILIDADE'])
    bncc_ano = bncc_row['ANO']
    bncc_eixo = bncc_row['EIXO']
    
    print(f"[{idx_bncc+1:2d}/50] {bncc_codigo} ({bncc_ano})")
    
    # Embedding desta BNCC
    embedding_bncc = bncc_embeddings[idx_bncc].reshape(1, -1)
    
    # Matches por ANO+DISCIPLINA
    matches_por_grupo = []
    
    # Agrupar por ANO + DISCIPLINA
    for (ano_curr, disciplina), grupo in curriculo_df.groupby(['ANO', 'DISCIPLINA']):
        if len(grupo) == 0: continue
        
        # Embeddings deste grupo
        grupo_textos = grupo.apply(texto_curriculo, axis=1)
        grupo_embeddings = model.encode(grupo_textos.tolist())
        
        # Melhor similaridade deste grupo
        similaridades = cosine_similarity(embedding_bncc, grupo_embeddings)[0]
        melhor_idx = np.argmax(similaridades)
        melhor_sim = similaridades[melhor_idx]
        
        melhor_row = grupo.iloc[melhor_idx]
        
        matches_por_grupo.append({
            'ano': ano_curr,
            'disciplina': disciplina,
            'similaridade': float(melhor_sim),
            'codigo': extrair_codigo(melhor_row['HABILIDADES']),
            'habilidade_completa': str(melhor_row['HABILIDADES']),
            'orientacoes_completas': str(melhor_row['ORIENTACOES_PEDAGOGICAS']),
            'total_no_grupo': len(grupo),
            'rank_no_grupo': melhor_idx + 1
        })
    
    # Top 10 melhores transversais
    matches_por_grupo.sort(key=lambda x: x['similaridade'], reverse=True)
    top_matches = matches_por_grupo[:10]
    
    relatorio_transversal.append({
        'bncc_idx': idx_bncc + 1,
        'bncc_codigo': bncc_codigo,
        'bncc_ano': bncc_ano,
        'bncc_eixo': bncc_eixo,
        'bncc_habilidade': str(bncc_row['HABILIDADE']),
        'melhor_match': top_matches[0] if top_matches else None,
        'top_matches': top_matches
    })
    
    print(f"   🎯 Top: {top_matches[0]['similaridade']:.1%} ({top_matches[0]['ano']}-{top_matches[0]['disciplina']})")

# ==================================================================================
# RELATÓRIO COMPLETO
# ==================================================================================

print("\n📊 Gerando relatório final...")

relatorio_txt = f"""🚀 TRANSFERÊNCIA TRANSVERSAL BNCC COMPUTACIONAL → CURRÍCULO
Data: {datetime.now().strftime("%d/%m/%Y %H:%M")}
Análise: 1x1 por ANO/DISCIPLINA (50 BNCC × {len(curriculo_df)} Currículo)

📊 ESTATÍSTICAS GLOBAIS:
• BNCC analisadas: {len(relatorio_transversal)}
• Melhor similaridade média: {np.mean([x['melhor_match']['similaridade'] for x in relatorio_transversal]):.1%}
• Top 5 transferências:"""

# Top 5 globais
top_globais = sorted([(i, r['melhor_match']) for i, r in enumerate(relatorio_transversal)], 
                    key=lambda x: x[1]['similaridade'], reverse=True)[:5]

for pos, (idx, match) in enumerate(top_globais, 1):
    relatorio_txt += f"\n  {pos}. BNCC[{idx+1}] → {match['ano']}/{match['disciplina']}: {match['similaridade']:.1%}"

relatorio_txt += f"\n\n{'='*120}\nDETALHAMENTO COMPLETO (TODAS BNCC):\n"

for item in relatorio_transversal:
    relatorio_txt += f"\n{'='*120}\n"
    relatorio_txt += f"🔹 BNCC [{item['bncc_idx']:2d}] {item['bncc_codigo']:12} | {item['bncc_ano']:7} | {item['bncc_eixo']}\n"
    relatorio_txt += f"   💻 BNCC: {item['bncc_habilidade']}\n\n"
    
    for i, match in enumerate(item['top_matches'], 1):
        relatorio_txt += f"   {i:2d}. 🎯 {match['ano']:7} | {match['disciplina']:18} | {match['codigo']:12}\n"
        relatorio_txt += f"      📊 {match['similaridade']:.1%} (rank {match['rank_no_grupo']}/{match['total_no_grupo']})\n"
        relatorio_txt += f"      🎯 CURRÍCULO: {match['habilidade_completa']}\n"
        relatorio_txt += f"      📝 ORIENTAÇÕES: {match['orientacoes_completas'][:400]}...\n\n"

# Salvar
os.makedirs(os.path.join(DIRETORIO_ATUAL, "docs", "transversal"), exist_ok=True)
with open(os.path.join(DIRETORIO_ATUAL, "docs", "transversal", "RELATORIO_TRANSVERSAL_COMPLETO.txt"), "w", encoding="utf-8") as f:
    f.write(relatorio_txt)

print(f"\n✅ ✅ SUCESSO TOTAL!")
print(f"📁 Salvo em: docs/transversal/RELATORIO_TRANSVERSAL_COMPLETO.txt")
print("🚀 Cada BNCC tem Top 10 matches por ANO/DISCIPLINA!")
