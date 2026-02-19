import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ==================================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ==================================================================================

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

# Arquivos dos ANOS FINAIS
CAMINHO_BNCC = os.path.join(DIRETORIO_ATUAL, "bncc_df_anosfinais.xlsx")
CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "Anos_Finais_Curriculo_Normalizado.xlsx")

# Fallback se o nome for diferente
if not os.path.exists(CAMINHO_CURRICULO):
    CAMINHO_CURRICULO = os.path.join(DIRETORIO_ATUAL, "curriculo_anos_finais_orientacoes.xlsx")

# ==================================================================================
# 2. CARREGAMENTO DO MODELO (ROBUSTO)
# ==================================================================================
print("🔄 Carregando modelo SentenceTransformer...")

# Tenta carregar ignorando a pasta 'models' se ela estiver corrompida,
# priorizando o cache do sistema que já funciona para o outro script.
try:
    # 1. Tenta o nome padrão (busca no cache global do HuggingFace)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Modelo carregado do cache global!")
except:
    try:
        # 2. Tenta forçar modo offline
        model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        print("✅ Modelo carregado (Offline Mode)!")
    except:
        try:
            # 3. Tenta carregar da pasta local apenas se as anteriores falharem
            caminho_local = os.path.join(DIRETORIO_ATUAL, 'models')
            if os.path.exists(caminho_local):
                model = SentenceTransformer(caminho_local)
                print("✅ Modelo carregado da pasta 'models'!")
            else:
                raise Exception("Pasta models não encontrada")
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO: Não foi possível carregar o modelo.")
            print(f"Erro detalhado: {e}")
            print("DICA: Apague a pasta 'models' se ela estiver vazia ou corrompida.")
            exit()

print("🚀 Análise Transversal BNCC (Anos Finais) → Currículo")
print("="*80)

# ==================================================================================
# 3. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==================================================================================

try:
    bncc_df = pd.read_excel(CAMINHO_BNCC)
    curriculo_df = pd.read_excel(CAMINHO_CURRICULO)
    
    # Padronizar colunas
    bncc_df.columns = bncc_df.columns.str.strip().str.upper()
    curriculo_df.columns = curriculo_df.columns.str.strip().str.upper()

    # Converter colunas chave para string
    for df in [bncc_df, curriculo_df]:
        for col in ['ANO', 'HABILIDADE', 'HABILIDADES', 'EIXO', 'DISCIPLINA']:
            if col in df.columns:
                df[col] = df[col].astype(str)

    print(f"📊 BNCC Anos Finais: {len(bncc_df)} habilidades")
    print(f"📊 Currículo Municipal: {len(curriculo_df)} habilidades")

except Exception as e:
    print(f"❌ Erro ao ler arquivos Excel: {e}")
    exit()

# ==================================================================================
# 4. FUNÇÕES AUXILIARES
# ==================================================================================

def extrair_codigo(texto):
    if pd.isna(texto): return "(SEM_CODIGO)"
    texto = str(texto)
    match = re.search(r'\(([A-Z]{2}\d{2}[A-Z]{2}\d{2})\)', texto)
    if match: return f"({match.group(1)})"
    return f"({texto[:15]}...)"

def identificar_ano_bncc(codigo_completo):
    """Retorna '6', '7', '8', '9' ou 'FAIXA'"""
    texto = str(codigo_completo)
    match = re.search(r'EF(\d{2})', texto)
    if match:
        digitos = match.group(1)
        if digitos in ['06', '07', '08', '09']:
            return str(int(digitos)) # Retorna '6', '7'
    return 'FAIXA'

def texto_bncc(linha):
    return f"{linha.get('ANO','')} | {linha.get('EIXO','')} | {linha.get('HABILIDADE','')} | {linha.get('EXEMPLOS','')}"

def texto_curriculo(linha):
    return f"{linha.get('ANO','')} | {linha.get('DISCIPLINA','')} | {linha.get('HABILIDADES','')} | {linha.get('ORIENTACOES_PEDAGOGICAS','')}"

# ==================================================================================
# 5. EMBEDDINGS E ANÁLISE
# ==================================================================================

print("\n🔄 Gerando embeddings da BNCC...")
bncc_textos = bncc_df.apply(texto_bncc, axis=1)
bncc_embeddings = model.encode(bncc_textos.tolist(), show_progress_bar=True)

relatorio_transversal = []
print("\n🚀 TRANSFERÊNCIA TRANSVERSAL 1x1 (FILTRO POR ANO)...")

# Se não tiver coluna ANO no currículo, cria uma genérica
if 'ANO' not in curriculo_df.columns:
    curriculo_df['ANO'] = 'GERAL'

for idx_bncc, (_, bncc_row) in enumerate(bncc_df.iterrows()):
    
    raw_habilidade = bncc_row.get('HABILIDADE', '')
    bncc_codigo = extrair_codigo(raw_habilidade)
    bncc_ano_raw = bncc_row.get('ANO', '')
    bncc_eixo = bncc_row.get('EIXO', '')
    
    # Identifica se é 6º ano, 7º ano ou Faixa (69, 67, etc)
    ano_alvo = identificar_ano_bncc(bncc_codigo)
    
    if (idx_bncc + 1) % 10 == 0:
        print(f"[{idx_bncc+1:2d}/{len(bncc_df)}] {bncc_codigo} (Alvo: {ano_alvo})")
    
    embedding_bncc = bncc_embeddings[idx_bncc].reshape(1, -1)
    
    matches_por_grupo = []
    
    # Agrupa o currículo por ANO e DISCIPLINA
    # Ex: Grupo ("6", "Matemática"), Grupo ("6", "História")...
    colunas_agrupamento = ['ANO', 'DISCIPLINA'] if 'DISCIPLINA' in curriculo_df.columns else ['ANO']
    
    for nome_grupo, grupo in curriculo_df.groupby(colunas_agrupamento):
        if len(grupo) == 0: continue
        
        # Extrai Ano e Disciplina do Grupo
        if isinstance(nome_grupo, tuple):
            ano_curr = str(nome_grupo[0])
            disciplina_curr = str(nome_grupo[1])
        else:
            ano_curr = str(nome_grupo)
            disciplina_curr = "Geral"

        # --- FILTRO LÓGICO ---
        # Se BNCC é específica (ex: 6º), ignora grupos de outros anos.
        # Se BNCC é Faixa, aceita tudo.
        if ano_alvo != 'FAIXA':
            # Verifica se "6" está dentro de "6º Ano" ou "Year 6"
            if ano_alvo not in ano_curr:
                continue 

        # Gera embeddings deste grupo
        grupo_textos = grupo.apply(texto_curriculo, axis=1)
        grupo_embeddings = model.encode(grupo_textos.tolist())
        
        # Acha o melhor match DENTRO deste grupo
        similaridades = cosine_similarity(embedding_bncc, grupo_embeddings)[0]
        melhor_idx = np.argmax(similaridades)
        melhor_sim = similaridades[melhor_idx]
        
        melhor_row = grupo.iloc[melhor_idx]
        
        matches_por_grupo.append({
            'ano': ano_curr,
            'disciplina': disciplina_curr,
            'similaridade': float(melhor_sim),
            'codigo': extrair_codigo(melhor_row.get('HABILIDADES', '')),
            'habilidade_completa': str(melhor_row.get('HABILIDADES', '')),
            'orientacoes_completas': str(melhor_row.get('ORIENTACOES_PEDAGOGICAS', '')),
            'rank_no_grupo': melhor_idx + 1,
            'total_no_grupo': len(grupo)
        })
    
    # Ordena matches (As melhores disciplinas para essa habilidade)
    matches_por_grupo.sort(key=lambda x: x['similaridade'], reverse=True)
    top_matches = matches_por_grupo[:10]
    
    relatorio_transversal.append({
        'bncc_idx': idx_bncc + 1,
        'bncc_codigo': bncc_codigo,
        'bncc_ano': bncc_ano_raw,
        'bncc_eixo': bncc_eixo,
        'bncc_habilidade': raw_habilidade,
        'top_matches': top_matches,
        'melhor_match': top_matches[0] if top_matches else None
    })
    
    if top_matches:
        print(f"   🎯 Top: {top_matches[0]['similaridade']:.1%} ({top_matches[0]['disciplina']} - {top_matches[0]['ano']})")

# ==================================================================================
# 6. RELATÓRIO FINAL
# ==================================================================================

print("\n📊 Gerando relatório final...")

relatorio_txt = f"""🚀 RELATÓRIO TRANSVERSAL BNCC (ANOS FINAIS) → CURRÍCULO
Data: {datetime.now().strftime("%d/%m/%Y %H:%M")}
Análise: 1x1 filtrada por ANO (Habilidade BNCC Ano X compara apenas com Currículo Ano X)

📊 RESUMO DAS MELHORES CORRESPONDÊNCIAS:
"""

# Ordena pelo Top Match Global
top_globais = sorted([r for r in relatorio_transversal if r['melhor_match']], 
                    key=lambda x: x['melhor_match']['similaridade'], reverse=True)[:10]

for i, item in enumerate(top_globais, 1):
    m = item['melhor_match']
    relatorio_txt += f"\n {i:2d}. {item['bncc_codigo']} ↔ {m['disciplina']} ({m['ano']}): {m['similaridade']:.1%}"

relatorio_txt += f"\n\n{'='*120}\nDETALHAMENTO COMPLETO:\n"

for item in relatorio_transversal:
    relatorio_txt += f"\n{'='*120}\n"
    relatorio_txt += f"🔹 BNCC [{item['bncc_idx']:2d}] {item['bncc_codigo']:10} | {item['bncc_eixo']}\n"
    relatorio_txt += f"   📜 {item['bncc_habilidade']}\n\n"
    
    if not item['top_matches']:
        relatorio_txt += "   ⚠️ Nenhuma correspondência encontrada (Verifique filtros de ano).\n"
    
    for i, match in enumerate(item['top_matches'], 1):
        relatorio_txt += f"   {i:2d}. 🎯 {match['disciplina']:20} | Ano: {match['ano']:5} | {match['codigo']}\n"
        relatorio_txt += f"      📊 {match['similaridade']:.1%} (rank {match['rank_no_grupo']}/{match['total_no_grupo']} na disciplina)\n"
        relatorio_txt += f"      📄 {match['habilidade_completa']}\n"
        relatorio_txt += f"      📝 {match['orientacoes_completas'][:200]}...\n"
        relatorio_txt += "\n"

caminho_saida = os.path.join(DIRETORIO_ATUAL, "docs", "anos finais")
os.makedirs(caminho_saida, exist_ok=True)
arquivo_final = os.path.join(caminho_saida, "RELATORIO_ANOS_FINAIS_FINAL.txt")

with open(arquivo_final, "w", encoding="utf-8") as f:
    f.write(relatorio_txt)

print(f"\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
print(f"📄 Relatório salvo em: {arquivo_final}")