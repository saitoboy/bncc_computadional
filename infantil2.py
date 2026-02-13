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
#                   CONFIGURAÇÃO INTELIGENTE DE CAMINHOS
# ==================================================================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_BNCC = os.path.join(DIRETORIO_ATUAL, "bncc_df_inf.xlsx")
CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "CURRICULO_ESTADO_educacao_infantil.xlsx")

def encontrar_modelo_real(pasta_base):
    print(f"🔍 Rastreando modelo dentro de: {pasta_base}")
    if not os.path.exists(pasta_base):
        print(f"⚠️ Pasta 'models' não encontrada em {pasta_base}")
        return None
    for root, dirs, files in os.walk(pasta_base):
        if 'config.json' in files and ('pytorch_model.bin' in files or 'model.safetensors' in files):
            print(f"🎯 Modelo encontrado em: {root}")
            return root
    print("❌ Arquivos do modelo não encontrados.")
    return None

PASTA_MODELS_GERAL = os.path.join(DIRETORIO_ATUAL, 'models')
CAMINHO_FINAL_MODELO = encontrar_modelo_real(PASTA_MODELS_GERAL)

if CAMINHO_FINAL_MODELO is None:
    print("⚠️ Usando configuração de fallback...")
    CAMINHO_FINAL_MODELO = 'all-MiniLM-L6-v2'
    CACHE_FOLDER = PASTA_MODELS_GERAL
else:
    CACHE_FOLDER = None

CONFIGURACOES = {
    'NOTA_CORTE': 0.80,
    'ARQUIVO_BNCC': CAMINHO_BNCC,
    'ARQUIVO_CURRICULO': CAMINHO_CURRICULO,
    'MODELO_EMBEDDINGS': CAMINHO_FINAL_MODELO,
    'GERAR_HEATMAP': True,
}
CONFIGURACOES['PREFIXO_ARQUIVOS'] = f"corte_{int(CONFIGURACOES['NOTA_CORTE']*100)}pct_"

print("="*80)
print("🚀 ANALISADOR FINAL CORRIGIDO - EDUCAÇÃO INFANTIL")
print("="*80)

# ==================================================================================
#                   CARREGAMENTO DOS DADOS
# ==================================================================================
try:
    bncc_df_inf = pd.read_excel(CONFIGURACOES['ARQUIVO_BNCC'])
    curriculo_df_inf = pd.read_excel(CONFIGURACOES['ARQUIVO_CURRICULO'])
    
    # NORMALIZAR NOMES DE COLUNAS
    bncc_df_inf.columns = bncc_df_inf.columns.str.strip()
    curriculo_df_inf.columns = curriculo_df_inf.columns.str.strip()
    
    print(f"✅ Dados carregados: {len(bncc_df_inf)} BNCC, {len(curriculo_df_inf)} Currículo")
except Exception as e:
    print(f"❌ Erro ao carregar: {e}")
    raise

# ==================================================================================
#                   DETECÇÃO INTELIGENTE DE COLUNAS
# ==================================================================================
def detectar_colunas_debug(df, nome_df):
    print(f"\n🔍 DEBUG {nome_df}:")
    print(f"  Colunas: {df.columns.tolist()}")
    
    # DETECÇÃO BASEADA NO DEBUG ANTERIOR (SABEMOS A ESTRUTURA)
    if nome_df == "BNCC":
        return 'EIXO', 'OBJETIVO DE APRENDIZAGEM', 'EXEMPLOS'
    elif nome_df == "CURRÍCULO":
        return 'EIXO', 'OBJETIVO DE APRENDIZAGEM', 'EXEMPLOS'
    
    # Fallback genérico
    eixo = next((c for c in df.columns if any(x in c.upper() for x in ['EIXO', 'CAMPO'])), df.columns[1])
    habilidade = next((c for c in df.columns if 'OBJETIVO' in c.upper() or 'HABILIDADE' in c.upper()), df.columns[2])
    exemplos = next((c for c in df.columns if 'EXEMPLOS' in c.upper()), df.columns[-1])
    
    print(f"  🎯 Detectado: Eixo='{eixo}' | Habilidade='{habilidade}' | Exemplos='{exemplos}'")
    return eixo, habilidade, exemplos

print("\n🔍 DETECTANDO COLUNAS...")
eixo_bncc, hab_bncc, ex_bncc = detectar_colunas_debug(bncc_df_inf, "BNCC")
eixo_curr, hab_curr, ex_curr = detectar_colunas_debug(curriculo_df_inf, "CURRÍCULO")

print(f"\n✅ BNCC: Eixo='{eixo_bncc}' | Obj='{hab_bncc}'")
print(f"✅ Currículo: Eixo='{eixo_curr}' | Obj='{hab_curr}'")

# ==================================================================================
#                   FUNÇÕES AUXILIARES CORRIGIDAS
# ==================================================================================
def extrair_codigo(obj):
    if pd.isna(obj) or str(obj).strip() == '':
        return "(SEM_TEXTO)"
    
    obj_str = str(obj).strip()
    
    patterns = [
        r'\b(E[I|L]\d{2}[A-Z]{2}\d{2})\b',  # EI01PC01
        r'\(([A-Z]{2}\d{2}[A-Z]{2}\d{2})\)',
        r'\b([A-Z]+\d+[A-Z]*\d+)\b',
        r'\(([A-Z0-9]{6,12})\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, obj_str, re.IGNORECASE)
        if match:
            return f"({match.group(1)})"
    
    texto_curto = obj_str[:25].strip(' .,!?')
    return f"({texto_curto}...)"

def concat_features(df, col_eixo, col_hab, col_ex):
    eixo_text = df[col_eixo].fillna('SEM_EIXO').astype(str)
    hab_text = df[col_hab].fillna('SEM_OBJETIVO').astype(str)
    ex_text = df[col_ex].fillna('N/A').astype(str)
    return eixo_text + " | " + hab_text + " | " + ex_text

# ==================================================================================
#                   CARREGAR MODELO E EMBEDDINGS
# ==================================================================================
print("\n🤖 Carregando modelo...")
try:
    if CACHE_FOLDER:
        model = SentenceTransformer(CAMINHO_FINAL_MODELO, cache_folder=CACHE_FOLDER)
    else:
        model = SentenceTransformer(CAMINHO_FINAL_MODELO)
    print("✅ Modelo carregado!")
except Exception as e:
    print(f"❌ Erro modelo: {e}")
    raise

bncc_texts = concat_features(bncc_df_inf, eixo_bncc, hab_bncc, ex_bncc)
curriculo_texts = concat_features(curriculo_df_inf, eixo_curr, hab_curr, ex_curr)

print("🔄 Gerando embeddings...")
bncc_embeddings = model.encode(bncc_texts.tolist(), show_progress_bar=True)
curriculo_embeddings = model.encode(curriculo_texts.tolist(), show_progress_bar=True)

print("🔄 Calculando similaridades...")
grau_similaridade = cosine_similarity(bncc_embeddings, curriculo_embeddings)

# ==================================================================================
#                   ALGORITMO CORRIGIDO COM ITERROWS()
# ==================================================================================
def encontrar_similaridade_balanceada(grau_similaridade, bncc_df, curriculo_df, nota_corte_inicial):
    relatorio_completo = []
    codigos_ja_usados = set()
    
    # ✅ CORRIGIDO: Agrupar currículo com iterrows()
    campos_curriculo = {}
    for idx, row in curriculo_df.iterrows():
        campo = row[eixo_curr] if pd.notna(row[eixo_curr]) else 'GERAL'
        if campo not in campos_curriculo:
            campos_curriculo[campo] = []
        
        hab_texto = row[hab_curr]
        codigo = extrair_codigo(hab_texto)
        campos_curriculo[campo].append({'indice': idx, 'codigo': codigo, 'row': row})
    
    # ✅ CORRIGIDO: Processar BNCC com iterrows()
    for idx_bncc, row_bncc in bncc_df.iterrows():
        similaridades_bncc = grau_similaridade[idx_bncc]
        
        bncc_texto_hab = row_bncc[hab_bncc]
        bncc_codigo = extrair_codigo(bncc_texto_hab)
        bncc_eixo = row_bncc[eixo_bncc]
        bncc_exemplo_val = row_bncc.get(ex_bncc, 'N/A')
        
        print(f"DEBUG BNCC {idx_bncc+1}: '{bncc_texto_hab[:60]}...' → {bncc_codigo}")
        
        # Encontrar correspondências por campo
        correspondencias_por_campo = {}
        for campo, objetivos_curr in campos_curriculo.items():
            melhores_do_campo = []
            for obj_curr in objetivos_curr:
                idx_curr = obj_curr['indice']
                codigo_curr = obj_curr['codigo']
                if codigo_curr in codigos_ja_usados:
                    continue
                
                sim = similaridades_bncc[idx_curr]
                melhores_do_campo.append({
                    'indice': idx_curr,
                    'codigo': codigo_curr,
                    'similaridade': sim,
                    'row': obj_curr['row'],
                    'campo': campo
                })
            if melhores_do_campo:
                melhores_do_campo.sort(key=lambda x: x['similaridade'], reverse=True)
                correspondencias_por_campo[campo] = melhores_do_campo
        
        # Lógica de matching
        habilidades_similares = []
        nota_corte_usada = nota_corte_inicial
        
        # Estratégia 1: Melhor de cada campo ≥ corte inicial
        for campo, candidatos in correspondencias_por_campo.items():
            if candidatos and candidatos[0]['similaridade'] >= nota_corte_inicial:
                melhor = candidatos[0]
                habilidades_similares.append(melhor)
                codigos_ja_usados.add(melhor['codigo'])
        
        # Estratégia 2: Reduzir corte adaptativamente
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
        
        # Estratégia 3: Melhor geral
        if not habilidades_similares:
            todas = []
            for cp, cands in correspondencias_por_campo.items():
                for c in cands:
                    if c['codigo'] not in codigos_ja_usados:
                        todas.append(c)
            if todas:
                best = max(todas, key=lambda x: x['similaridade'])
                habilidades_similares.append(best)
                codigos_ja_usados.add(best['codigo'])
                nota_corte_usada = best['similaridade']
        
        habilidades_similares.sort(key=lambda x: x['similaridade'], reverse=True)
        
        # Criar item do relatório
        item_relatorio = {
            'bncc_indice': idx_bncc + 1,
            'bncc_codigo': bncc_codigo,
            'bncc_eixo': bncc_eixo,
            'bncc_objetivo': str(bncc_texto_hab)[:250] + "..." if len(str(bncc_texto_hab)) > 250 else str(bncc_texto_hab),
            'bncc_exemplos': str(bncc_exemplo_val)[:100],
            'habilidades_similares': [],
            'tem_similaridade_original': any(h['similaridade'] >= nota_corte_inicial for h in habilidades_similares),
            'nota_corte_usada': nota_corte_usada,
            'quantidade_similares': len(habilidades_similares),
            'maior_similaridade': max([h['similaridade'] for h in habilidades_similares]) if habilidades_similares else 0
        }
        
        # Adicionar correspondências
        for sim in habilidades_similares:
            row_c = sim['row']
            curr_hab_txt = row_c[hab_curr]
            curr_ex_txt = row_c.get(ex_curr, 'N/A')
            
            item_relatorio['habilidades_similares'].append({
                'curriculo_indice': sim['indice'] + 1,
                'curriculo_codigo': sim['codigo'],
                'curriculo_eixo': sim['campo'],
                'curriculo_objetivo': str(curr_hab_txt)[:250] + "..." if len(str(curr_hab_txt)) > 250 else str(curr_hab_txt),
                'curriculo_exemplos': str(curr_ex_txt)[:100],
                'similaridade': sim['similaridade']
            })
        
        relatorio_completo.append(item_relatorio)
    
    return relatorio_completo

# ==================================================================================
#                   EXECUÇÃO PRINCIPAL
# ==================================================================================
NOTA_CORTE = CONFIGURACOES['NOTA_CORTE']
data_relatorio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"\n📊 Gerando relatório (Corte: {NOTA_CORTE*100}%)...")
relatorio_completo = encontrar_similaridade_balanceada(grau_similaridade, bncc_df_inf, curriculo_df_inf, NOTA_CORTE)

# Estatísticas
estatisticas = {
    'total_bncc': len(bncc_df_inf),
    'bncc_com_similaridade_original': sum(1 for hab in relatorio_completo if hab['tem_similaridade_original']),
    'total_matches_acima_corte': sum(1 for hab in relatorio_completo for sim in hab['habilidades_similares'] if sim['similaridade'] >= NOTA_CORTE),
    'notas_corte_usadas': [hab['nota_corte_usada'] for hab in relatorio_completo],
    'nota_corte_original': NOTA_CORTE,
    'modelo_usado': os.path.basename(str(CAMINHO_FINAL_MODELO)),
    'data_analise': data_relatorio
}

print("✅ Análise concluída!")

# ==================================================================================
#                   SALVAMENTO
# ==================================================================================
docs_path = os.path.join(DIRETORIO_ATUAL, "docs", "infantil")
os.makedirs(docs_path, exist_ok=True)

prefixo = CONFIGURACOES['PREFIXO_ARQUIVOS']

def gerar_relatorio_texto():
    txt = f"""RELATÓRIO FINAL CORRIGIDO - BNCC X CURRÍCULO INFANTIL
Data: {estatisticas['data_analise']}
Corte Inicial: {estatisticas['nota_corte_original']:.0%}
Modelo: {estatisticas['modelo_usado']}

RESUMO ESTATÍSTICO:
• Total BNCC: {estatisticas['total_bncc']}
• Matches ≥80%: {estatisticas['bncc_com_similaridade_original']}
• Total Matches Fortes: {estatisticas['total_matches_acima_corte']}
• Média Corte Usada: {np.mean(estatisticas['notas_corte_usadas']):.1%}
• Maior Similaridade: {max([hab['maior_similaridade'] for hab in relatorio_completo]):.1%}

DETALHAMENTO POR HABILIDADE BNCC:
"""
    
    for hab in relatorio_completo:
        txt += f"\n{'='*90}\n"
        txt += f"BNCC #{hab['bncc_indice']:2d}: {hab['bncc_codigo']:12s} | {hab['bncc_eixo']}\n"
        txt += f"OBJETIVO BNCC: {hab['bncc_objetivo']}\n"
        txt += f"ANÁLISE: Corte={hab['nota_corte_usada']:.1%} | Matches={hab['quantidade_similares']} | Melhor={hab['maior_similaridade']:.1%}\n"
        
        if hab['habilidades_similares']:
            for s in hab['habilidades_similares']:
                txt += f"  → CURR #{s['curriculo_indice']:3d}: {s['curriculo_codigo']:10s} ({s['curriculo_eixo']})\n"
                txt += f"     SIM={s['similaridade']:.1%} | OBJETIVO: {s['curriculo_objetivo']}\n"
        else:
            txt += "  ⚠️ NENHUMA CORRESPONDÊNCIA ENCONTRADA\n"
    
    return txt

# SALVAR ARQUIVOS
try:
    # Relatório principal
    with open(os.path.join(docs_path, f"infantil_{prefixo}relatorio.txt"), "w", encoding="utf-8") as f:
        f.write(gerar_relatorio_texto())
    
    # CSV para análise
    dados = []
    for h in relatorio_completo:
        for s in h['habilidades_similares']:
            dados.append({
                'BNCC_Codigo': h['bncc_codigo'],
                'BNCC_Eixo': h['bncc_eixo'],
                'BNCC_Objeto': h['bncc_objetivo'],
                'Curr_Codigo': s['curriculo_codigo'],
                'Curr_Eixo': s['curriculo_eixo'],
                'Curr_Objeto': s['curriculo_objetivo'],
                'Similaridade': s['similaridade'],
                'Corte_Usado': h['nota_corte_usada']
            })
    df_csv = pd.DataFrame(dados)
    df_csv.to_csv(os.path.join(docs_path, f"infantil_{prefixo}dados.csv"), index=False, encoding="utf-8-sig")
    
    # Resumo executivo
    with open(os.path.join(docs_path, f"infantil_{prefixo}resumo.txt"), "w", encoding="utf-8") as f:
        f.write(f"RESUMO EXECUTIVO - BNCC INFANTIL vs CURRÍCULO\n{data_relatorio}\n\n")
        f.write(f"Total BNCC analisada: {len(bncc_df_inf)}\n")
        f.write(f"Cobertura 80%+: {estatisticas['bncc_com_similaridade_original']} ({estatisticas['bncc_com_similaridade_original']/len(bncc_df_inf)*100:.1f}%)\n")
        f.write(f"\nTOP 10 MELHORES MATCHES:\n")
        top_matches = df_csv.nlargest(10, 'Similaridade')
        for idx, row in top_matches.iterrows():
            f.write(f"{row['BNCC_Codigo']:<12} ↔ {row['Curr_Codigo']:<12}: {row['Similaridade']:.1%}\n")
    
    print("✅ Arquivos salvos:")
    print(f"   📄 {docs_path}/infantil_{prefixo}relatorio.txt")
    print(f"   📊 {docs_path}/infantil_{prefixo}dados.csv")
    print(f"   📋 {docs_path}/infantil_{prefixo}resumo.txt")
    
except Exception as e:
    print(f"❌ Erro ao salvar: {e}")

# HEATMAP CORRIGIDO
if CONFIGURACOES['GERAR_HEATMAP']:
    try:
        bncc_cods = [extrair_codigo(text) for text in bncc_df_inf[hab_bncc]]
        curr_cods = [extrair_codigo(text) for text in curriculo_df_inf[hab_curr][:min(50, len(curriculo_df_inf))]]
        
        plt.figure(figsize=(20, 12))
        sim_subset = grau_similaridade[:, :len(curr_cods)]
        sns.heatmap(sim_subset, 
                   xticklabels=curr_cods, 
                   yticklabels=bncc_cods,
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues_r',
                   cbar_kws={'label': 'Similaridade'})
        plt.title('MATRIZ DE SIMILARIDADE BNCC vs CURRÍCULO\n(Valores mais escuros = maior similaridade)')
        plt.xlabel('Currículo Estadual')
        plt.ylabel('BNCC Computação')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(docs_path, f"infantil_{prefixo}heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print("✅ Heatmap salvo!")
    except Exception as e:
        print(f"⚠️ Heatmap: {e}")

print("\n" + "="*80)
print("🎉 ANÁLISE 100% FINALIZADA!")
print("📂 Veja resultados em: docs/infantil/")
print("="*80)
