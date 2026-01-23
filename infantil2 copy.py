import os
import sys
from dotenv import load_dotenv

# Adicionar o diretório raiz ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Carrega variáveis do .env da raiz do projeto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Garante que matplotlib só salve imagens, sem abrir janelas
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Importar funções do módulo de similaridade refatorado
from core.similarity import (
    extrair_codigo,
    concat_features_bncc,
    concat_features_curriculo,
    encontrar_similaridade_balanceada,
    carregar_modelo_embeddings,
    gerar_embeddings,
    calcular_similaridades
)

print("🔧 Módulo infantil carregado com algoritmo balanceado")

# ==================================================================================
#                           CONFIGURAÇÕES DINÂMICAS
# ==================================================================================
CONFIGURACOES = {
    'NOTA_CORTE': 0.80,
    'ARQUIVO_BNCC': "bncc_df_inf.xlsx",
    'ARQUIVO_CURRICULO': "curriculo_df_inf.xlsx",
    'MODELO_EMBEDDINGS': 'all-MiniLM-L6-v2',
    'MOSTRAR_TOP_MATCHES_TERMINAL': 10,
    'MOSTRAR_TOP_CORRESPONDENCIAS': 15,
    'GERAR_HEATMAP': True,
    'TAMANHO_HEATMAP': (20, 20),
}

CONFIGURACOES['PREFIXO_ARQUIVOS'] = f"corte_{int(CONFIGURACOES['NOTA_CORTE']*100)}pct_"

print("="*80)
print("🚀 ANALISADOR DE SIMILARIDADE BNCC x CURRÍCULO MUNICIPAL - INFANTIL")
print("🎯 ALGORITMO BALANCEADO - EVITA DUPLICATAS E DISTRIBUI POR DISCIPLINAS")
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

# Normalizar os nomes das colunas para evitar problemas de espaço/acentuação
curriculo_df_inf.columns = curriculo_df_inf.columns.str.strip()
bncc_df_inf.columns = bncc_df_inf.columns.str.strip()

# Normalizar nomes das disciplinas (EIXOS) no currículo para evitar duplicatas
def normalizar_disciplina(nome):
    """Normaliza nomes de disciplinas removendo variações desnecessárias"""
    if pd.isna(nome):
        return "SEM_DISCIPLINA"
    
    nome_str = str(nome).strip().upper()
    
    # Mapear variações para nomes consistentes
    mapeamento = {
        "ESPAÇOS, TEMPOS, QUANTIDADES E RELAÇÕES": "ESPAÇOS, TEMPOS, QUANTIDADES, RELAÇÕES",
        "ESCUTA, FALA, PENSAMENTOS E IMAGINAÇÃO": "ESCUTA, FALA, PENSAMENTO E IMAGINAÇÃO",
        "TRAÇOS, SONS, CORES E FROMAS": "TRAÇOS, SONS, CORES E FORMAS",
        "O EU, O OUTRO E O NÓS": "O EU, O OUTRO E O NÓS"
    }
    
    return mapeamento.get(nome_str, nome_str)

# Aplicar normalização
print("� Normalizando nomes das disciplinas/eixos...")
bncc_df_inf['EIXO'] = bncc_df_inf['EIXO'].apply(normalizar_disciplina)
curriculo_df_inf['EIXO'] = curriculo_df_inf['EIXO'].apply(normalizar_disciplina)

print("📊 Disciplinas BNCC após normalização:", bncc_df_inf['EIXO'].unique())
print("📊 Disciplinas Currículo após normalização:", curriculo_df_inf['EIXO'].unique())

# Carregar modelo e gerar embeddings usando o módulo refatorado
print("🤖 Carregando modelo de embeddings...")
model = carregar_modelo_embeddings(CONFIGURACOES['MODELO_EMBEDDINGS'])

print("🔄 Gerando embeddings...")
bncc_texts = concat_features_bncc(bncc_df_inf)
curriculo_texts = concat_features_curriculo(curriculo_df_inf)

bncc_embeddings = gerar_embeddings(model, bncc_texts)
curriculo_embeddings = gerar_embeddings(model, curriculo_texts)

print("🔄 Calculando similaridades...")
grau_similaridade = calcular_similaridades(bncc_embeddings, curriculo_embeddings)

# ==================================================================================
#                           ANÁLISE COM ALGORITMO BALANCEADO
# ==================================================================================

NOTA_CORTE = CONFIGURACOES['NOTA_CORTE']
data_relatorio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"📊 Aplicando algoritmo balanceado com nota de corte inicial de {NOTA_CORTE*100}%...")
print("🎯 Garantindo distribuição equilibrada por disciplinas e evitando duplicatas...")

# DEBUG: Verificar primeiras linhas dos dados antes do algoritmo
print(f"\n🔍 DEBUG - VERIFICANDO ESTRUTURA DOS DADOS:")
print(f"📊 BNCC - Primeiras 2 linhas da coluna 'OBJETIVO DE APRENDIZAGEM':")
for i, (idx, linha) in enumerate(bncc_df_inf.head(2).iterrows()):
    obj_aprendizagem = linha['OBJETIVO DE APRENDIZAGEM']
    codigo_extraido = extrair_codigo(obj_aprendizagem)
    print(f"   {i+1}. {str(obj_aprendizagem)[:60]}...")
    print(f"      Código extraído: {codigo_extraido}")

print(f"\n📊 CURRÍCULO - Primeiras 2 linhas da coluna 'OBJETIVO DE APRENDIZAGEM':")
for i, (idx, linha) in enumerate(curriculo_df_inf.head(2).iterrows()):
    obj_aprendizagem = linha['OBJETIVO DE APRENDIZAGEM']
    codigo_extraido = extrair_codigo(obj_aprendizagem)
    print(f"   {i+1}. {str(obj_aprendizagem)[:60]}...")
    print(f"      Código extraído: {codigo_extraido}")

print(f"\n🎯 Iniciando algoritmo balanceado...")

# Usar o algoritmo balanceado do módulo de similaridade
relatorio_completo = encontrar_similaridade_balanceada(
    grau_similaridade, 
    bncc_df_inf, 
    curriculo_df_inf, 
    NOTA_CORTE
)

# Calcular estatísticas
estatisticas = {
    'total_bncc': len(bncc_df_inf),
    'bncc_com_similaridade_original': sum(1 for h in relatorio_completo if h['tem_similaridade_original']),
    'bncc_com_similaridade_balanceada': len(relatorio_completo),
    'total_matches_acima_corte': sum(len([s for s in h['habilidades_similares'] if s['similaridade'] >= NOTA_CORTE]) for h in relatorio_completo),
    'nota_corte_original': NOTA_CORTE,
    'modelo_usado': CONFIGURACOES['MODELO_EMBEDDINGS'],
    'data_analise': data_relatorio,
    'habilidades_unicas_usadas': len(set(s['curriculo_codigo'] for h in relatorio_completo for s in h['habilidades_similares'])),
    'distribuicao_por_disciplina': {}
}

# Calcular distribuição por disciplina
for habilidade in relatorio_completo:
    for similar in habilidade['habilidades_similares']:
        disciplina = similar.get('disciplina', similar.get('curriculo_eixo', 'SEM_DISCIPLINA'))
        estatisticas['distribuicao_por_disciplina'][disciplina] = (
            estatisticas['distribuicao_por_disciplina'].get(disciplina, 0) + 1
        )

print("✅ Algoritmo balanceado aplicado! Gerando relatórios...")

# Estatísticas do algoritmo balanceado
print(f"📈 Resultados do algoritmo balanceado:")
print(f"   🎯 {estatisticas['bncc_com_similaridade_original']}/{estatisticas['total_bncc']} habilidades com nota de corte original")
print(f"   🔄 {estatisticas['bncc_com_similaridade_balanceada']}/{estatisticas['total_bncc']} habilidades com correspondências (balanceado)")
print(f"   ✨ {estatisticas['habilidades_unicas_usadas']} habilidades únicas do currículo utilizadas")
print(f"   📊 Distribuição por disciplina: {estatisticas['distribuicao_por_disciplina']}")

# Verificar se os códigos estão sendo extraídos corretamente
print("\n🔍 VERIFICAÇÃO DOS CÓDIGOS EXTRAÍDOS (ALGORITMO BALANCEADO):")
print("-" * 50)
for i in range(min(3, len(relatorio_completo))):
    hab = relatorio_completo[i]
    print(f"BNCC {i+1}: {hab['bncc_codigo']}")
    print(f"Eixo: {hab['bncc_eixo']}")
    print(f"Correspondências encontradas: {len(hab['habilidades_similares'])}")
    if hab['habilidades_similares']:
        melhor = hab['habilidades_similares'][0]
        print(f"Melhor match: {melhor['curriculo_codigo']} ({melhor['similaridade']:.1%})")
        print(f"Disciplina: {melhor.get('disciplina', melhor.get('curriculo_eixo', 'N/A'))}")
    print("-" * 30)

# ==================================================================================
#                           FUNÇÕES DE GERAÇÃO DE RELATÓRIOS
# ==================================================================================

def gerar_relatorio_texto():
    relatorio_txt = f"""
==================================================================================
                    RELATÓRIO DE SIMILARIDADE BNCC x CURRÍCULO MUNICIPAL - INFANTIL
                                  ALGORITMO BALANCEADO
==================================================================================
Data do relatório: {estatisticas['data_analise']}
Nota de corte inicial: {estatisticas['nota_corte_original']*100}% de similaridade
Modelo utilizado: {estatisticas['modelo_usado']}
Algoritmo: BALANCEADO (evita duplicatas e distribui por disciplinas)

ESTATÍSTICAS GERAIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total de habilidades BNCC analisadas: {estatisticas['total_bncc']}
• Habilidades BNCC com similaridade ≥ {estatisticas['nota_corte_original']*100}% (nota original): {estatisticas['bncc_com_similaridade_original']} ({estatisticas['bncc_com_similaridade_original']/estatisticas['total_bncc']*100:.1f}%)
• Habilidades BNCC com correspondência (algoritmo balanceado): {estatisticas['bncc_com_similaridade_balanceada']} ({estatisticas['bncc_com_similaridade_balanceada']/estatisticas['total_bncc']*100:.1f}%)
• Total de matches acima da nota de corte original: {estatisticas['total_matches_acima_corte']}
• Habilidades únicas do currículo utilizadas: {estatisticas['habilidades_unicas_usadas']}
• Taxa de aproveitamento das habilidades do currículo: {estatisticas['habilidades_unicas_usadas']/len(curriculo_df_inf)*100:.1f}%

DISTRIBUIÇÃO POR DISCIPLINA/EIXO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for disciplina, count in estatisticas['distribuicao_por_disciplina'].items():
        relatorio_txt += f"• {disciplina}: {count} correspondências\n"
    
    relatorio_txt += f"""
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

ALGORITMO USADO: {'Original' if habilidade['tem_similaridade_original'] else 'Balanceado'}
QUANTIDADE DE HABILIDADES SIMILARES: {len(habilidade['habilidades_similares'])}

"""
        
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
    
    return relatorio_txt

def gerar_relatorio_csv():
    dados_csv = []
    
    for habilidade in relatorio_completo:
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
                'Algoritmo_Usado': 'Original' if habilidade['tem_similaridade_original'] else 'Balanceado',
                'Acima_Nota_Corte': 'Sim' if similar['similaridade'] >= NOTA_CORTE else 'Não',
                'Disciplina': similar.get('disciplina', similar.get('curriculo_eixo', 'N/A')),
                'Data_Analise': estatisticas['data_analise']
            })
    
    return pd.DataFrame(dados_csv)

def gerar_resumo_executivo():
    resumo = f"""
==================================================================================
                            RESUMO EXECUTIVO - INFANTIL
                                  ALGORITMO BALANCEADO
==================================================================================
Data: {estatisticas['data_analise']}
Nota de corte inicial: {estatisticas['nota_corte_original']*100}%
Algoritmo: BALANCEADO (evita duplicatas e distribui por disciplinas)
Modelo: {estatisticas['modelo_usado']}

PRINCIPAIS DESCOBERTAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• {estatisticas['bncc_com_similaridade_original']} de {estatisticas['total_bncc']} habilidades da BNCC ({estatisticas['bncc_com_similaridade_original']/estatisticas['total_bncc']*100:.1f}%) têm correspondência com nota de corte original ≥ {estatisticas['nota_corte_original']*100}%

• {estatisticas['bncc_com_similaridade_balanceada']} de {estatisticas['total_bncc']} habilidades da BNCC ({estatisticas['bncc_com_similaridade_balanceada']/estatisticas['total_bncc']*100:.1f}%) têm correspondência usando algoritmo balanceado

• Total de {estatisticas['total_matches_acima_corte']} conexões identificadas acima da nota de corte original

• {estatisticas['habilidades_unicas_usadas']} habilidades únicas do currículo utilizadas de {len(curriculo_df_inf)} disponíveis ({estatisticas['habilidades_unicas_usadas']/len(curriculo_df_inf)*100:.1f}% de aproveitamento)

DISTRIBUIÇÃO POR DISCIPLINA/EIXO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for disciplina, count in estatisticas['distribuicao_por_disciplina'].items():
        percentual = count / sum(estatisticas['distribuicao_por_disciplina'].values()) * 100
        resumo += f"\n• {disciplina}: {count} correspondências ({percentual:.1f}%)"
    
    resumo += f"""

HABILIDADES BNCC COM MAIOR NÚMERO DE CORRESPONDÊNCIAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    top_correspondencias = sorted(relatorio_completo, key=lambda x: len(x['habilidades_similares']), reverse=True)[:CONFIGURACOES['MOSTRAR_TOP_CORRESPONDENCIAS']]
    
    for i, hab in enumerate(top_correspondencias, 1):
        resumo += f"\n{i:2d}. {hab['bncc_codigo']} - {len(hab['habilidades_similares'])} correspondências"
        resumo += f"\n    {hab['bncc_eixo']}"
        resumo += f"\n    Algoritmo: {'Original' if hab['tem_similaridade_original'] else 'Balanceado'}"
        if hab['habilidades_similares']:
            resumo += f"\n    Máx. similaridade: {max(s['similaridade'] for s in hab['habilidades_similares']):.1%}"
    
    resumo += f"\n\nHABILIDADES QUE USARAM ALGORITMO BALANCEADO:\n"
    resumo += "━" * 70 + "\n"
    
    algoritmo_balanceado = [h for h in relatorio_completo if not h['tem_similaridade_original']]
    for hab in algoritmo_balanceado:
        resumo += f"\n• {hab['bncc_codigo']} - {hab['bncc_eixo']}"
        if hab['habilidades_similares']:
            resumo += f"\n  Máx. similaridade: {max(s['similaridade'] for s in hab['habilidades_similares']):.1%}"
        resumo += f"\n  {hab['bncc_objetivo'][:100]}{'...' if len(hab['bncc_objetivo']) > 100 else ''}\n"
    
    resumo += f"""

VANTAGENS DO ALGORITMO BALANCEADO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Evita duplicação de habilidades do currículo
• Garante distribuição equilibrada entre disciplinas/eixos
• Maximiza o aproveitamento das habilidades disponíveis no currículo
• Oferece correspondências mais diversificadas para planejamento pedagógico
• Reduz a concentração em poucas habilidades do currículo
"""
    
    return resumo

# ==================================================================================
#                           SALVAMENTO DOS ARQUIVOS
# ==================================================================================

# ==================================================================================
#                           SALVAMENTO DOS ARQUIVOS
# ==================================================================================

# Criar pasta docs/infantil se não existir
docs_path = os.path.join("docs", "infantil")
if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    print(f"📁 Pasta '{docs_path}' criada para organizar os relatórios")

nome_relatorio_completo = os.path.join(docs_path, f"infantil_{CONFIGURACOES['PREFIXO_ARQUIVOS']}relatorio_completo.txt")
nome_relatorio_csv = os.path.join(docs_path, f"infantil_{CONFIGURACOES['PREFIXO_ARQUIVOS']}relatorio.csv")
nome_resumo = os.path.join(docs_path, f"infantil_{CONFIGURACOES['PREFIXO_ARQUIVOS']}resumo_executivo.txt")
nome_heatmap = os.path.join(docs_path, f"infantil_{CONFIGURACOES['PREFIXO_ARQUIVOS']}heatmap_similaridade.png")

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
print("📊 RESUMO DA ANÁLISE - ALGORITMO BALANCEADO (INFANTIL):")
print("="*80)
print(f"✅ Análise concluída com sucesso!")
print(f"⚙️  Nota de corte inicial: {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"📊 {estatisticas['bncc_com_similaridade_original']}/{estatisticas['total_bncc']} habilidades BNCC têm correspondência ≥ {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"🎯 {estatisticas['bncc_com_similaridade_balanceada']}/{estatisticas['total_bncc']} habilidades BNCC têm correspondência com algoritmo balanceado")
print(f"🔍 {estatisticas['total_matches_acima_corte']} conexões identificadas acima da nota de corte original")
print(f"✨ {estatisticas['habilidades_unicas_usadas']}/{len(curriculo_df_inf)} habilidades únicas do currículo utilizadas ({estatisticas['habilidades_unicas_usadas']/len(curriculo_df_inf)*100:.1f}%)")

print(f"\n� DISTRIBUIÇÃO POR DISCIPLINA/EIXO:")
for disciplina, count in estatisticas['distribuicao_por_disciplina'].items():
    percentual = count / sum(estatisticas['distribuicao_por_disciplina'].values()) * 100
    print(f"   • {disciplina}: {count} correspondências ({percentual:.1f}%)")

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
            'curriculo_eixo': similar['curriculo_eixo'],
            'disciplina': similar.get('disciplina', similar.get('curriculo_eixo', 'N/A'))
        })

top_matches = sorted(todos_matches, key=lambda x: x['similaridade'], reverse=True)[:CONFIGURACOES['MOSTRAR_TOP_MATCHES_TERMINAL']]

for i, match in enumerate(top_matches, 1):
    print(f"{i:2d}. {match['bncc_codigo']} ↔ {match['curriculo_codigo']} | {match['similaridade']:.1%}")
    print(f"    BNCC: {match['bncc_eixo']}")
    print(f"    Currículo: {match['disciplina']}")
    print()

print("="*80)
print("🎯 ALGORITMO BALANCEADO ATIVO:")
print("   • Evita duplicação de habilidades do currículo")
print("   • Garante distribuição equilibrada por disciplinas")
print("   • Maximiza aproveitamento das habilidades disponíveis")
print("💡 Para alterar a nota de corte, modifique CONFIGURACOES['NOTA_CORTE']")
print("📁 Consulte os arquivos de relatório na pasta docs/infantil/ para análise completa!")
print("="*80)