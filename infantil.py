import os
from dotenv import load_dotenv

# Carrega variáveis do .env da raiz do projeto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')  # Garante que matplotlib só salve imagens, sem abrir janelas
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================================================================================
#                           CONFIGURAÇÕES DINÂMICAS
# ==================================================================================
CONFIGURACOES = {
    'NOTA_CORTE': 0.60,
    'ARQUIVO_BNCC': "bncc_df_inf.xlsx",
    'ARQUIVO_CURRICULO': "curriculo_df_inf.xlsx",
    'MODELO_EMBEDDINGS': 'all-MiniLM-L6-v2',
    'MOSTRAR_TOP_MATCHES_TERMINAL': 10,
    'MOSTRAR_TOP_CORRESPONDENCIAS': 15,
    'PREFIXO_ARQUIVOS': f"corte_{int(0.60*100)}pct_",
    'GERAR_HEATMAP': True,
    'TAMANHO_HEATMAP': (20, 20),
}

CONFIGURACOES['PREFIXO_ARQUIVOS'] = f"corte_{int(CONFIGURACOES['NOTA_CORTE']*100)}pct_"

print("="*80)
print("🚀 ANALISADOR DE SIMILARIDADE BNCC x CURRÍCULO MUNICIPAL")
print("="*80)
print(f"⚙️  CONFIGURAÇÕES ATIVAS:")
print(f"   📊 Nota de corte: {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"   🤖 Modelo: {CONFIGURACOES['MODELO_EMBEDDINGS']}")
print(f"   📁 Prefixo arquivos: {CONFIGURACOES['PREFIXO_ARQUIVOS']}")
print("="*80)

# ==================================================================================
#                           PROCESSAMENTO DOS DADOS
# ==================================================================================

# Carregamento dos dados
try:
    bncc_df_inf = pd.read_excel(CONFIGURACOES['ARQUIVO_BNCC'])
    curriculo_df_inf = pd.read_excel(CONFIGURACOES['ARQUIVO_CURRICULO'])
    print(f"✅ Dados carregados: {len(bncc_df_inf)} habilidades BNCC, {len(curriculo_df_inf)} habilidades currículo")
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    raise

# Verificação das colunas necessárias nos DataFrames
def checar_colunas(df, nome_df):
    colunas_necessarias = ['EIXO', 'OBJETIVO DE APRENDIZAGEM', 'EXEMPLOS']
    for col in colunas_necessarias:
        assert col in df.columns, f"{nome_df} deve ter a coluna '{col}'"
    print(f"✅ Colunas verificadas em {nome_df}")

checar_colunas(bncc_df_inf, 'BNCC')
checar_colunas(curriculo_df_inf, 'Currículo')

# Normaliza os nomes das colunas para evitar problemas de espaço/acentuação
curriculo_df_inf.columns = curriculo_df_inf.columns.str.strip()
bncc_df_inf.columns = bncc_df_inf.columns.str.strip()

# ==================================================================================
#                    FUNÇÃO CORRIGIDA PARA EXTRAIR CÓDIGOS
# ==================================================================================

def extrair_codigo(obj):
    """
    Extrai códigos no formato (EI03CO01), (EF01CO01), etc.
    """
    if pd.isna(obj):
        return "(SEM_COD)"
    
    obj_str = str(obj)
    
    # Padrão mais robusto para capturar códigos BNCC
    patterns = [
        r'\(([A-Z]{2}\d{2}[A-Z]{2}\d{2})\)',  # Padrão principal: EI03CO01
        r'\(([A-Z]+\d+[A-Z]*\d*)\)',          # Padrão mais geral
        r'\(([A-Z0-9]+)\)',                   # Padrão ainda mais geral
    ]
    
    for pattern in patterns:
        match = re.search(pattern, obj_str)
        if match:
            return f"({match.group(1)})"
    
    # Se não encontrou nenhum padrão, tenta extrair qualquer coisa entre parênteses no início
    match_inicio = re.search(r'^\(([^)]+)\)', obj_str)
    if match_inicio:
        return f"({match_inicio.group(1)})"
    
    # Como último recurso, pega os primeiros 15 caracteres + "..."
    return f"({obj_str[:15]}...)" if len(obj_str) > 15 else f"({obj_str})"

# Testar a função de extração com alguns exemplos
print("\n🔍 TESTE DA FUNÇÃO DE EXTRAÇÃO DE CÓDIGOS:")
print("-" * 50)
for i in range(min(5, len(bncc_df_inf))):
    obj_original = bncc_df_inf['OBJETIVO DE APRENDIZAGEM'].iloc[i]
    codigo_extraido = extrair_codigo(obj_original)
    print(f"Original: {obj_original}")
    print(f"Código:   {codigo_extraido}")
    print("-" * 30)

# Função para concatenar as features relevantes
def concat_features(df):
    return (
        df['EIXO'].astype(str) + " | " +
        df['OBJETIVO DE APRENDIZAGEM'].astype(str) + " | " +
        df['EXEMPLOS'].astype(str)
    )

bncc_texts = concat_features(bncc_df_inf)
curriculo_texts = concat_features(curriculo_df_inf)

# Carregar modelo de embeddings
try:
    if 'HUGGINGFACE_HUB_DISABLE_SYMLINKS' in os.environ:
        del os.environ['HUGGINGFACE_HUB_DISABLE_SYMLINKS']
    
    print(f"🤖 Carregando modelo {CONFIGURACOES['MODELO_EMBEDDINGS']}...")
    model = SentenceTransformer(CONFIGURACOES['MODELO_EMBEDDINGS'])
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    print("💡 Tente rodar: pip install sentence-transformers")
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

NOTA_CORTE = CONFIGURACOES['NOTA_CORTE']
data_relatorio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"📊 Gerando relatório com nota de corte de {NOTA_CORTE*100}%...")

# Criar relatório detalhado
relatorio_completo = []
estatisticas = {
    'total_bncc': len(bncc_df_inf),
    'bncc_com_similaridade': 0,
    'bncc_sem_similaridade': 0,
    'total_matches_acima_corte': 0,
    'nota_corte': NOTA_CORTE,
    'modelo_usado': CONFIGURACOES['MODELO_EMBEDDINGS'],
    'data_analise': data_relatorio
}

for idx_bncc, linha_bncc in enumerate(bncc_df_inf.itertuples(index=False)):
    # Obter similaridades desta habilidade BNCC com todas do currículo
    similaridades_bncc = grau_similaridade[idx_bncc]
    
    # Encontrar habilidades do currículo acima da nota de corte
    indices_similares = np.where(similaridades_bncc >= NOTA_CORTE)[0]
    
    # CORREÇÃO: Usar o índice direto do DataFrame em vez de itertuples()
    objetivo_aprendizagem = bncc_df_inf['OBJETIVO DE APRENDIZAGEM'].iloc[idx_bncc]
    bncc_codigo = extrair_codigo(objetivo_aprendizagem)
    
    habilidade_bncc = {
        'bncc_indice': idx_bncc + 1,
        'bncc_codigo': bncc_codigo,
        'bncc_eixo': linha_bncc.EIXO,
        'bncc_objetivo': objetivo_aprendizagem,
        'bncc_exemplos': linha_bncc.EXEMPLOS,
        'habilidades_similares': [],
        'tem_similaridade': len(indices_similares) > 0,
        'quantidade_similares': len(indices_similares),
        'maior_similaridade': np.max(similaridades_bncc) if len(similaridades_bncc) > 0 else 0
    }
    
    # Adicionar habilidades similares do currículo
    for idx_similar in indices_similares:
        linha_curriculo = curriculo_df_inf.iloc[idx_similar]
        curriculo_codigo = extrair_codigo(linha_curriculo['OBJETIVO DE APRENDIZAGEM'])
        
        habilidade_similar = {
            'curriculo_indice': idx_similar + 1,
            'curriculo_codigo': curriculo_codigo,
            'curriculo_eixo': linha_curriculo['EIXO'],
            'curriculo_objetivo': linha_curriculo['OBJETIVO DE APRENDIZAGEM'],
            'curriculo_exemplos': linha_curriculo['EXEMPLOS'],
            'similaridade': similaridades_bncc[idx_similar]
        }
        habilidade_bncc['habilidades_similares'].append(habilidade_similar)
        estatisticas['total_matches_acima_corte'] += 1
    
    # Ordenar por similaridade (maior primeiro)
    habilidade_bncc['habilidades_similares'].sort(key=lambda x: x['similaridade'], reverse=True)
    
    relatorio_completo.append(habilidade_bncc)
    
    # Atualizar estatísticas
    if habilidade_bncc['tem_similaridade']:
        estatisticas['bncc_com_similaridade'] += 1
    else:
        estatisticas['bncc_sem_similaridade'] += 1

print("✅ Relatório analisado! Salvando arquivos...")

# Verificar se os códigos estão sendo extraídos corretamente
print("\n🔍 VERIFICAÇÃO FINAL DOS CÓDIGOS EXTRAÍDOS:")
print("-" * 50)
for i in range(min(3, len(relatorio_completo))):
    hab = relatorio_completo[i]
    print(f"BNCC {i+1}: {hab['bncc_codigo']}")
    print(f"Objetivo: {hab['bncc_objetivo'][:80]}...")
    print(f"Similaridades encontradas: {hab['quantidade_similares']}")
    if hab['habilidades_similares']:
        print(f"Melhor match: {hab['habilidades_similares'][0]['curriculo_codigo']} ({hab['habilidades_similares'][0]['similaridade']:.1%})")
    print("-" * 30)

# ==================================================================================
#                           FUNÇÕES DE GERAÇÃO DE RELATÓRIOS
# ==================================================================================

def gerar_relatorio_texto():
    relatorio_txt = f"""
==================================================================================
                    RELATÓRIO DE SIMILARIDADE BNCC x CURRÍCULO MUNICIPAL
==================================================================================
Data do relatório: {estatisticas['data_analise']}
Nota de corte: {estatisticas['nota_corte']*100}% de similaridade
Modelo utilizado: {estatisticas['modelo_usado']}

ESTATÍSTICAS GERAIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total de habilidades BNCC analisadas: {estatisticas['total_bncc']}
• Habilidades BNCC com similaridade ≥ {estatisticas['nota_corte']*100}%: {estatisticas['bncc_com_similaridade']} ({estatisticas['bncc_com_similaridade']/estatisticas['total_bncc']*100:.1f}%)
• Habilidades BNCC sem similaridade ≥ {estatisticas['nota_corte']*100}%: {estatisticas['bncc_sem_similaridade']} ({estatisticas['bncc_sem_similaridade']/estatisticas['total_bncc']*100:.1f}%)
• Total de matches acima da nota de corte: {estatisticas['total_matches_acima_corte']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELATÓRIO DETALHADO POR HABILIDADE BNCC:

"""
    
    for idx, habilidade in enumerate(relatorio_completo):
        relatorio_txt += f"""
{'='*80}
HABILIDADE BNCC #{habilidade['bncc_indice']} - {habilidade['bncc_codigo']}
{'='*80}

EIXO BNCC: {habilidade['bncc_eixo']}

OBJETIVO BNCC: {habilidade['bncc_objetivo']}

EXEMPLOS BNCC: {habilidade['bncc_exemplos']}

QUANTIDADE DE HABILIDADES SIMILARES (≥ {estatisticas['nota_corte']*100}%): {habilidade['quantidade_similares']}
MAIOR SIMILARIDADE ENCONTRADA: {habilidade['maior_similaridade']:.1%}

"""
        
        if habilidade['tem_similaridade']:
            relatorio_txt += "HABILIDADES SIMILARES DO CURRÍCULO:\n"
            relatorio_txt += "-" * 60 + "\n"
            
            for i, similar in enumerate(habilidade['habilidades_similares'], 1):
                relatorio_txt += f"""
{i}. CURRÍCULO #{similar['curriculo_indice']} - {similar['curriculo_codigo']} | SIMILARIDADE: {similar['similaridade']:.1%}
   
   EIXO CURRÍCULO: {similar['curriculo_eixo']}
   
   OBJETIVO CURRÍCULO: {similar['curriculo_objetivo']}
   
   EXEMPLOS CURRÍCULO: {similar['curriculo_exemplos']}
   
   {'─' * 50}
"""
        else:
            relatorio_txt += f"❌ NENHUMA HABILIDADE DO CURRÍCULO ATINGE A NOTA DE CORTE DE {estatisticas['nota_corte']*100}%\n"
            relatorio_txt += f"   (Maior similaridade encontrada: {habilidade['maior_similaridade']:.1%})\n"
    
    return relatorio_txt

def gerar_relatorio_csv():
    dados_csv = []
    
    for habilidade in relatorio_completo:
        if habilidade['tem_similaridade']:
            for similar in habilidade['habilidades_similares']:
                dados_csv.append({
                    'BNCC_Indice': habilidade['bncc_indice'],
                    'BNCC_Codigo': habilidade['bncc_codigo'],
                    'BNCC_Eixo': habilidade['bncc_eixo'],
                    'BNCC_Objetivo': habilidade['bncc_objetivo'],
                    'BNCC_Exemplos': habilidade['bncc_exemplos'],
                    'Curriculo_Indice': similar['curriculo_indice'],
                    'Curriculo_Codigo': similar['curriculo_codigo'],
                    'Curriculo_Eixo': similar['curriculo_eixo'],
                    'Curriculo_Objetivo': similar['curriculo_objetivo'],
                    'Curriculo_Exemplos': similar['curriculo_exemplos'],
                    'Similaridade': similar['similaridade'],
                    'Similaridade_Percentual': f"{similar['similaridade']:.1%}",
                    'Nota_Corte_Usada': f"{estatisticas['nota_corte']*100}%",
                    'Data_Analise': estatisticas['data_analise']
                })
        else:
            dados_csv.append({
                'BNCC_Indice': habilidade['bncc_indice'],
                'BNCC_Codigo': habilidade['bncc_codigo'],
                'BNCC_Eixo': habilidade['bncc_eixo'],
                'BNCC_Objetivo': habilidade['bncc_objetivo'],
                'BNCC_Exemplos': habilidade['bncc_exemplos'],
                'Curriculo_Indice': 'N/A',
                'Curriculo_Codigo': 'N/A',
                'Curriculo_Eixo': 'N/A',
                'Curriculo_Objetivo': f'Nenhuma similaridade ≥ {estatisticas["nota_corte"]*100}%',
                'Curriculo_Exemplos': 'N/A',
                'Similaridade': habilidade['maior_similaridade'],
                'Similaridade_Percentual': f"{habilidade['maior_similaridade']:.1%}",
                'Nota_Corte_Usada': f"{estatisticas['nota_corte']*100}%",
                'Data_Analise': estatisticas['data_analise']
            })
    
    return pd.DataFrame(dados_csv)

def gerar_resumo_executivo():
    resumo = f"""
==================================================================================
                            RESUMO EXECUTIVO
==================================================================================
Data: {estatisticas['data_analise']}
Nota de corte: {estatisticas['nota_corte']*100}%
Modelo: {estatisticas['modelo_usado']}

PRINCIPAIS DESCOBERTAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• {estatisticas['bncc_com_similaridade']} de {estatisticas['total_bncc']} habilidades da BNCC ({estatisticas['bncc_com_similaridade']/estatisticas['total_bncc']*100:.1f}%) têm pelo menos uma correspondência no currículo atual com similaridade ≥ {estatisticas['nota_corte']*100}%

• {estatisticas['bncc_sem_similaridade']} habilidades da BNCC ({estatisticas['bncc_sem_similaridade']/estatisticas['total_bncc']*100:.1f}%) NÃO possuem correspondência adequada no currículo atual

• Total de {estatisticas['total_matches_acima_corte']} conexões identificadas acima da nota de corte

HABILIDADES BNCC COM MAIOR NÚMERO DE CORRESPONDÊNCIAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    top_correspondencias = sorted(relatorio_completo, key=lambda x: x['quantidade_similares'], reverse=True)[:CONFIGURACOES['MOSTRAR_TOP_CORRESPONDENCIAS']]
    
    for i, hab in enumerate(top_correspondencias, 1):
        resumo += f"\n{i:2d}. {hab['bncc_codigo']} - {hab['quantidade_similares']} correspondências (máx: {hab['maior_similaridade']:.1%})"
        resumo += f"\n    {hab['bncc_eixo']}"
    
    resumo += f"\n\nHABILIDADES BNCC SEM CORRESPONDÊNCIA (precisam ser implementadas):\n"
    resumo += "━" * 70 + "\n"
    
    sem_correspondencia = [h for h in relatorio_completo if not h['tem_similaridade']]
    for hab in sem_correspondencia:
        resumo += f"\n• {hab['bncc_codigo']} - {hab['bncc_eixo']}"
        resumo += f"\n  Máx. similaridade: {hab['maior_similaridade']:.1%}"
        resumo += f"\n  {hab['bncc_objetivo'][:100]}{'...' if len(hab['bncc_objetivo']) > 100 else ''}\n"
    
    return resumo

# ==================================================================================
#                           SALVAMENTO DOS ARQUIVOS
# ==================================================================================

nome_relatorio_completo = f"{CONFIGURACOES['PREFIXO_ARQUIVOS']}relatorio_completo.txt"
nome_relatorio_csv = f"{CONFIGURACOES['PREFIXO_ARQUIVOS']}relatorio.csv"
nome_resumo = f"{CONFIGURACOES['PREFIXO_ARQUIVOS']}resumo_executivo.txt"
nome_heatmap = f"{CONFIGURACOES['PREFIXO_ARQUIVOS']}heatmap_similaridade.png"

try:
    # Relatório completo em texto
    with open(nome_relatorio_completo, "w", encoding="utf-8") as f:
        f.write(gerar_relatorio_texto())
    
    # Relatório em CSV
    df_csv = gerar_relatorio_csv()
    df_csv.to_csv(nome_relatorio_csv, index=False, encoding="utf-8-sig")
    
    # Resumo executivo
    with open(nome_resumo, "w", encoding="utf-8") as f:
        f.write(gerar_resumo_executivo())
    
    print("✅ Relatórios salvos com sucesso!")
    print(f"   📄 {nome_relatorio_completo} - Relatório detalhado completo")
    print(f"   📊 {nome_relatorio_csv} - Dados em formato CSV")
    print(f"   📋 {nome_resumo} - Resumo com principais descobertas")
    
except Exception as e:
    print(f"❌ Erro ao salvar relatórios: {e}")

# Gerar heatmap se configurado
if CONFIGURACOES['GERAR_HEATMAP']:
    try:
        # Extrai códigos para usar como índices
        bncc_codigos = bncc_df_inf['OBJETIVO DE APRENDIZAGEM'].apply(extrair_codigo).tolist()
        curriculo_codigos = curriculo_df_inf['OBJETIVO DE APRENDIZAGEM'].apply(extrair_codigo).tolist()
        
        # Cria DataFrame de similaridade
        sim_df = pd.DataFrame(grau_similaridade, 
                              index=bncc_codigos,
                              columns=curriculo_codigos)
        
        # Gera heatmap com tamanho configurável
        tamanho_h, tamanho_c = CONFIGURACOES['TAMANHO_HEATMAP']
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            sim_df.iloc[:tamanho_h, :tamanho_c],
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0, vmax=1,
            cbar_kws={'label': 'Similaridade'}
        )
        plt.title(f"Heatmap de Similaridade BNCC x Currículo Municipal\nNota de corte: {CONFIGURACOES['NOTA_CORTE']*100}%")
        plt.xlabel("Currículo (códigos)")
        plt.ylabel("BNCC (códigos)")
        plt.tight_layout()
        plt.savefig(nome_heatmap, dpi=300, bbox_inches='tight')
        print(f"   🎨 {nome_heatmap} - Heatmap de similaridade")
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível gerar o heatmap: {e}")

# ==================================================================================
#                           EXIBIÇÃO NO TERMINAL
# ==================================================================================

# Exibir resumo no terminal
print("\n" + "="*80)
print("📊 RESUMO DA ANÁLISE:")
print("="*80)
print(f"✅ Análise concluída com sucesso!")
print(f"⚙️  Nota de corte utilizada: {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"📊 {estatisticas['bncc_com_similaridade']}/{estatisticas['total_bncc']} habilidades BNCC têm correspondência ≥ {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"🔍 {estatisticas['total_matches_acima_corte']} conexões identificadas")
print(f"⚠️  {estatisticas['bncc_sem_similaridade']} habilidades BNCC precisam ser implementadas")

# Mostrar algumas das melhores correspondências no terminal
print(f"\n🏆 TOP {CONFIGURACOES['MOSTRAR_TOP_MATCHES_TERMINAL']} MELHORES CORRESPONDÊNCIAS:")
print("-" * 70)

todos_matches = []
for hab in relatorio_completo:
    for similar in hab['habilidades_similares']:
        todos_matches.append({
            'bncc_codigo': hab['bncc_codigo'],
            'curriculo_codigo': similar['curriculo_codigo'],
            'similaridade': similar['similaridade'],
            'bncc_eixo': hab['bncc_eixo'],
            'curriculo_eixo': similar['curriculo_eixo']
        })

top_matches = sorted(todos_matches, key=lambda x: x['similaridade'], reverse=True)[:CONFIGURACOES['MOSTRAR_TOP_MATCHES_TERMINAL']]

for i, match in enumerate(top_matches, 1):
    print(f"{i:2d}. {match['bncc_codigo']} ↔ {match['curriculo_codigo']} | {match['similaridade']:.1%}")
    print(f"    BNCC: {match['bncc_eixo']}")
    print(f"    Currículo: {match['curriculo_eixo']}")
    print()

print("="*80)
print("💡 DICA: Para alterar a nota de corte, modifique CONFIGURACOES['NOTA_CORTE'] no topo do script!")
print("📁 Consulte os arquivos de relatório para análise completa!")
print("="*80)