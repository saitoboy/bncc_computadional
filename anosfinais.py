import os
from dotenv import load_dotenv

# Carrega variáveis do .env da raiz do projeto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# ==================================================================================
#                           CONFIGURAÇÃO DE PROXY
# ==================================================================================
# Configurar proxy para conexões externas (Hugging Face)
proxy_config = {
    'http': 'http://guilherme.saito:890484gS@10.10.30.9:3128',
    'https': 'http://guilherme.saito:890484gS@10.10.30.9:3128'  # Note: usando http:// mesmo para https
}

# Configurar variáveis de ambiente para o proxy
os.environ['HTTP_PROXY'] = proxy_config['http']
os.environ['HTTPS_PROXY'] = proxy_config['https']
os.environ['http_proxy'] = proxy_config['http']
os.environ['https_proxy'] = proxy_config['https']

# Configurar requests para usar proxy
import requests
requests.adapters.DEFAULT_RETRIES = 3

print("🔧 Proxy configurado para acessar Hugging Face")

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
    'NOTA_CORTE': 0.80,
    'ARQUIVO_BNCC': "bncc_df_anosfinais.xlsx",
    'ARQUIVO_CURRICULO': "Anos_Finais_Curriculo_Normalizado.xlsx",
    'MODELO_EMBEDDINGS': 'all-MiniLM-L6-v2',
    'MOSTRAR_TOP_MATCHES_TERMINAL': 10,
    'MOSTRAR_TOP_CORRESPONDENCIAS': 15,
    'PREFIXO_ARQUIVOS': f"corte_{int(0.80*100)}pct_",
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
    obj_original = bncc_df_inf['HABILIDADE'].iloc[i]
    codigo_extraido = extrair_codigo(obj_original)
    print(f"Original: {obj_original}")
    print(f"Código:   {codigo_extraido}")
    print("-" * 30)

# Função para concatenar as features relevantes
def concat_features_bncc(df):
    return (
        df['EIXO'].astype(str) + " | " +
        df['HABILIDADE'].astype(str) + " | " +
        df['EXEMPLOS'].astype(str)
    )

def concat_features_curriculo(df):
    return (
        df['DISCIPLINA'].astype(str) + " | " +
        df['HABILIDADES'].astype(str) + " | " +
        df['ORIENTACOES_PEDAGOGICAS'].astype(str)
    )

bncc_texts = concat_features_bncc(bncc_df_inf)
curriculo_texts = concat_features_curriculo(curriculo_df_inf)

# Carregar modelo de embeddings
try:
    # Configurações adicionais para contornar problemas de proxy/SSL
    if 'HUGGINGFACE_HUB_DISABLE_SYMLINKS' in os.environ:
        del os.environ['HUGGINGFACE_HUB_DISABLE_SYMLINKS']
    
    # Desabilitar verificação SSL se necessário (apenas para ambientes corporativos)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Configurar timeout maior para downloads
    os.environ['HF_HUB_TIMEOUT'] = '120'
    
    print(f"🤖 Carregando modelo {CONFIGURACOES['MODELO_EMBEDDINGS']}...")
    print("📡 Conectando através do proxy corporativo...")
    
    model = SentenceTransformer(CONFIGURACOES['MODELO_EMBEDDINGS'])
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    print("💡 Tentativas de solução:")
    print("   1. Verificar se o proxy está configurado corretamente")
    print("   2. Tentar usar o modelo offline se já foi baixado antes")
    print("   3. Verificar conectividade com a internet")
    
    # Tentar carregar um modelo local ou menor como fallback
    try:
        print("🔄 Tentando modelo alternativo...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Modelo alternativo carregado com sucesso!")
    except:
        raise Exception("Não foi possível carregar nenhum modelo. Verifique a configuração do proxy.")

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
#                           ANÁLISE E RELATÓRIOS COM BUSCA ADAPTATIVA
# ==================================================================================

def encontrar_similaridade_balanceada(grau_similaridade, bncc_df, curriculo_df, nota_corte_inicial):
    """
    Encontra similaridades balanceadas por disciplina, evitando duplicatas
    """
    relatorio_completo = []
    habilidades_ja_usadas = set()  # Rastrear códigos já utilizados
    
    # Agrupar currículo por disciplina
    disciplinas_curriculo = {}
    for idx, linha in enumerate(curriculo_df.itertuples(index=False)):
        disciplina = getattr(linha, 'DISCIPLINA', getattr(linha, 'EIXO', 'SEM_DISCIPLINA'))
        if disciplina not in disciplinas_curriculo:
            disciplinas_curriculo[disciplina] = []
        
        # Extrair código da habilidade do currículo
        if hasattr(linha, 'HABILIDADES'):
            codigo = extrair_codigo(linha.HABILIDADES)
        elif hasattr(linha, 'OBJETIVO_DE_APRENDIZAGEM'):
            codigo = extrair_codigo(linha.OBJETIVO_DE_APRENDIZAGEM)
        else:
            codigo = f"CURR_{idx}"
            
        disciplinas_curriculo[disciplina].append({
            'indice': idx,
            'codigo': codigo,
            'linha': linha
        })
    
    print(f"🎯 Disciplinas encontradas: {list(disciplinas_curriculo.keys())}")
    print(f"📊 Distribuição por disciplina: {[(d, len(h)) for d, h in disciplinas_curriculo.items()]}")
    
    # Para cada habilidade BNCC
    for idx_bncc, linha_bncc in enumerate(bncc_df.itertuples(index=False)):
        similaridades_bncc = grau_similaridade[idx_bncc]
        
        # Extrair código BNCC
        if hasattr(linha_bncc, 'HABILIDADE'):
            bncc_codigo = extrair_codigo(linha_bncc.HABILIDADE)
            bncc_objetivo = linha_bncc.HABILIDADE
        elif hasattr(linha_bncc, 'OBJETIVO_DE_APRENDIZAGEM'):
            bncc_codigo = extrair_codigo(linha_bncc.OBJETIVO_DE_APRENDIZAGEM)
            bncc_objetivo = linha_bncc.OBJETIVO_DE_APRENDIZAGEM
        else:
            bncc_codigo = f"BNCC_{idx_bncc}"
            bncc_objetivo = "OBJETIVO NÃO ENCONTRADO"
        
        # Buscar melhores correspondências por disciplina
        correspondencias_por_disciplina = {}
        
        for disciplina, habilidades_disc in disciplinas_curriculo.items():
            melhores_da_disciplina = []
            
            for hab_curriculo in habilidades_disc:
                idx_curr = hab_curriculo['indice']
                codigo_curr = hab_curriculo['codigo']
                
                # Pular se já foi usado
                if codigo_curr in habilidades_ja_usadas:
                    continue
                
                similaridade = similaridades_bncc[idx_curr]
                melhores_da_disciplina.append({
                    'indice': idx_curr,
                    'codigo': codigo_curr,
                    'similaridade': similaridade,
                    'linha': hab_curriculo['linha'],
                    'disciplina': disciplina
                })
            
            # Ordenar por similaridade (maior primeiro)
            melhores_da_disciplina.sort(key=lambda x: x['similaridade'], reverse=True)
            correspondencias_por_disciplina[disciplina] = melhores_da_disciplina
        
        # Aplicar estratégia de distribuição balanceada
        habilidades_similares = []
        nota_corte_usada = nota_corte_inicial
        
        # ESTRATÉGIA 1: Tentar pegar a melhor de cada disciplina com nota de corte original
        for disciplina, candidatos in correspondencias_por_disciplina.items():
            if candidatos and candidatos[0]['similaridade'] >= nota_corte_inicial:
                melhor = candidatos[0]
                habilidades_similares.append(melhor)
                habilidades_ja_usadas.add(melhor['codigo'])
        
        # ESTRATÉGIA 2: Se não conseguiu nenhuma, usar busca adaptativa
        if not habilidades_similares:
            nota_corte_atual = nota_corte_inicial
            
            while not habilidades_similares and nota_corte_atual > 0.1:
                nota_corte_atual -= 0.01
                
                # Tentar pegar pelo menos uma de cada disciplina
                for disciplina, candidatos in correspondencias_por_disciplina.items():
                    for candidato in candidatos:
                        if (candidato['similaridade'] >= nota_corte_atual and 
                            candidato['codigo'] not in habilidades_ja_usadas):
                            habilidades_similares.append(candidato)
                            habilidades_ja_usadas.add(candidato['codigo'])
                            break  # Só uma por disciplina
                
                if habilidades_similares:
                    nota_corte_usada = nota_corte_atual
                    break
        
        # ESTRATÉGIA 3: Se ainda não tem nada, pegar pelo menos a melhor geral disponível
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
        
        # ESTRATÉGIA 4: Adicionar mais correspondências se houver espaço (máximo 3 por habilidade BNCC)
        if len(habilidades_similares) < 3:
            for disciplina, candidatos in correspondencias_por_disciplina.items():
                if len(habilidades_similares) >= 3:
                    break
                    
                for candidato in candidatos[1:]:  # Pular o primeiro (já foi considerado)
                    if (candidato['codigo'] not in habilidades_ja_usadas and 
                        candidato['similaridade'] >= nota_corte_usada * 0.9):  # 90% da nota de corte usada
                        habilidades_similares.append(candidato)
                        habilidades_ja_usadas.add(candidato['codigo'])
                        break
        
        # Ordenar por similaridade
        habilidades_similares.sort(key=lambda x: x['similaridade'], reverse=True)
        
        # Montar estrutura do relatório
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
        
        # Adicionar detalhes das habilidades similares
        for similar in habilidades_similares:
            linha_curriculo = similar['linha']
            
            # Detectar coluna de habilidade do currículo
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
        
        # Log de progresso
        if (idx_bncc + 1) % 10 == 0:
            print(f"📈 Processadas {idx_bncc + 1}/{len(bncc_df)} habilidades BNCC")
    
    # Estatísticas finais
    total_habilidades_usadas = len(habilidades_ja_usadas)
    total_habilidades_curriculo = len(curriculo_df)
    
    print(f"\n📊 ESTATÍSTICAS DE DISTRIBUIÇÃO:")
    print(f"   🎯 Habilidades do currículo utilizadas: {total_habilidades_usadas}/{total_habilidades_curriculo} ({total_habilidades_usadas/total_habilidades_curriculo*100:.1f}%)")
    print(f"   🚫 Habilidades não utilizadas: {total_habilidades_curriculo - total_habilidades_usadas}")
    
    # Estatísticas por disciplina
    disciplinas_usadas = {}
    for hab_bncc in relatorio_completo:
        for similar in hab_bncc['habilidades_similares']:
            disc = similar['curriculo_eixo']
            disciplinas_usadas[disc] = disciplinas_usadas.get(disc, 0) + 1
    
    print(f"   📚 Distribuição de uso por disciplina:")
    for disc, count in sorted(disciplinas_usadas.items()):
        total_disc = len(disciplinas_curriculo.get(disc, []))
        percentual = count/total_disc*100 if total_disc > 0 else 0
        print(f"      {disc}: {count}/{total_disc} ({percentual:.1f}%)")
    
    return relatorio_completo

def encontrar_similaridade_adaptativa(similaridades_bncc, nota_corte_inicial):
    """
    Função mantida para compatibilidade - DEPRECADA
    Use encontrar_similaridade_balanceada() para melhor distribuição
    """
    nota_corte_atual = nota_corte_inicial
    indices_similares = np.where(similaridades_bncc >= nota_corte_atual)[0]
    
    # Se não encontrou nada com a nota de corte inicial, vai diminuindo
    while len(indices_similares) == 0 and nota_corte_atual > 0.1:
        nota_corte_atual -= 0.01  # Diminui 1% por vez
        indices_similares = np.where(similaridades_bncc >= nota_corte_atual)[0]
    
    # Se ainda não encontrou nada, pega pelo menos a melhor (mais similar)
    if len(indices_similares) == 0:
        idx_melhor = np.argmax(similaridades_bncc)
        indices_similares = np.array([idx_melhor])
        nota_corte_atual = similaridades_bncc[idx_melhor]
    
    return indices_similares, nota_corte_atual

NOTA_CORTE = CONFIGURACOES['NOTA_CORTE']
data_relatorio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"📊 Gerando relatório com nota de corte inicial de {NOTA_CORTE*100}%...")
print("🔄 Usando busca adaptativa para garantir pelo menos uma correspondência por habilidade...")

# Criar relatório detalhado
relatorio_completo = []
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

for idx_bncc, linha_bncc in enumerate(bncc_df_inf.itertuples(index=False)):
    # Obter similaridades desta habilidade BNCC com todas do currículo
    similaridades_bncc = grau_similaridade[idx_bncc]
    
    # Usar busca adaptativa para encontrar pelo menos uma correspondência
    indices_similares, nota_corte_usada = encontrar_similaridade_adaptativa(similaridades_bncc, NOTA_CORTE)
    estatisticas['notas_corte_usadas'].append(nota_corte_usada)
    
    # CORREÇÃO: Usar o índice direto do DataFrame
    objetivo_aprendizagem = bncc_df_inf['HABILIDADE'].iloc[idx_bncc]
    bncc_codigo = extrair_codigo(objetivo_aprendizagem)
    
    habilidade_bncc = {
        'bncc_indice': idx_bncc + 1,
        'bncc_codigo': bncc_codigo,
        'bncc_eixo': linha_bncc.EIXO,
        'bncc_objetivo': objetivo_aprendizagem,
        'bncc_exemplos': linha_bncc.EXEMPLOS,
        'habilidades_similares': [],
        'tem_similaridade_original': len(np.where(similaridades_bncc >= NOTA_CORTE)[0]) > 0,
        'nota_corte_usada': nota_corte_usada,
        'quantidade_similares': len(indices_similares),
        'maior_similaridade': np.max(similaridades_bncc) if len(similaridades_bncc) > 0 else 0
    }
    
    # Adicionar habilidades similares do currículo
    for idx_similar in indices_similares:
        linha_curriculo = curriculo_df_inf.iloc[idx_similar]
        curriculo_codigo = extrair_codigo(linha_curriculo['HABILIDADES'])
        
        habilidade_similar = {
            'curriculo_indice': idx_similar + 1,
            'curriculo_codigo': curriculo_codigo,
            'curriculo_eixo': linha_curriculo['DISCIPLINA'],
            'curriculo_objetivo': linha_curriculo['HABILIDADES'],
            'curriculo_exemplos': linha_curriculo['ORIENTACOES_PEDAGOGICAS'],
            'similaridade': similaridades_bncc[idx_similar]
        }
        habilidade_bncc['habilidades_similares'].append(habilidade_similar)
        
        # Contar apenas matches que atendem à nota de corte original
        if similaridades_bncc[idx_similar] >= NOTA_CORTE:
            estatisticas['total_matches_acima_corte'] += 1
    
    # Ordenar por similaridade (maior primeiro)
    habilidade_bncc['habilidades_similares'].sort(key=lambda x: x['similaridade'], reverse=True)
    
    relatorio_completo.append(habilidade_bncc)
    
    # Atualizar estatísticas
    if habilidade_bncc['tem_similaridade_original']:
        estatisticas['bncc_com_similaridade_original'] += 1
    
    # Todas as habilidades terão pelo menos uma correspondência devido à busca adaptativa
    estatisticas['bncc_com_similaridade_adaptativa'] += 1

print("✅ Relatório analisado! Salvando arquivos...")

# Estatísticas das notas de corte usadas
nota_corte_media = np.mean(estatisticas['notas_corte_usadas'])
nota_corte_min = np.min(estatisticas['notas_corte_usadas'])
nota_corte_max = np.max(estatisticas['notas_corte_usadas'])

print(f"📈 Estatísticas da busca adaptativa:")
print(f"   Nota de corte média usada: {nota_corte_media:.1%}")
print(f"   Nota de corte mínima: {nota_corte_min:.1%}")
print(f"   Nota de corte máxima: {nota_corte_max:.1%}")

# Verificar se os códigos estão sendo extraídos corretamente
print("\n🔍 VERIFICAÇÃO FINAL DOS CÓDIGOS EXTRAÍDOS:")
print("-" * 50)
for i in range(min(3, len(relatorio_completo))):
    hab = relatorio_completo[i]
    print(f"BNCC {i+1}: {hab['bncc_codigo']}")
    print(f"Objetivo: {hab['bncc_objetivo'][:80]}...")
    print(f"Nota de corte usada: {hab['nota_corte_usada']:.1%}")
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
Nota de corte inicial: {estatisticas['nota_corte_original']*100}% de similaridade
Modelo utilizado: {estatisticas['modelo_usado']}
Busca adaptativa: ATIVADA (garante pelo menos 1 correspondência por habilidade)

ESTATÍSTICAS GERAIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total de habilidades BNCC analisadas: {estatisticas['total_bncc']}
• Habilidades BNCC com similaridade ≥ {estatisticas['nota_corte_original']*100}% (nota original): {estatisticas['bncc_com_similaridade_original']} ({estatisticas['bncc_com_similaridade_original']/estatisticas['total_bncc']*100:.1f}%)
• Habilidades BNCC com correspondência (busca adaptativa): {estatisticas['bncc_com_similaridade_adaptativa']} ({estatisticas['bncc_com_similaridade_adaptativa']/estatisticas['total_bncc']*100:.1f}%)
• Total de matches acima da nota de corte original: {estatisticas['total_matches_acima_corte']}
• Nota de corte média usada: {np.mean(estatisticas['notas_corte_usadas']):.1%}
• Nota de corte mínima usada: {np.min(estatisticas['notas_corte_usadas']):.1%}
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

NOTA DE CORTE USADA: {habilidade['nota_corte_usada']:.1%} {'(original)' if habilidade['tem_similaridade_original'] else '(adaptativa)'}
QUANTIDADE DE HABILIDADES SIMILARES: {habilidade['quantidade_similares']}
MAIOR SIMILARIDADE ENCONTRADA: {habilidade['maior_similaridade']:.1%}

"""
        
        relatorio_txt += "HABILIDADES SIMILARES DO CURRÍCULO:\n"
        relatorio_txt += "-" * 60 + "\n"
        
        for i, similar in enumerate(habilidade['habilidades_similares'], 1):
            relatorio_txt += f"""
{i}. CURRÍCULO #{similar['curriculo_indice']} - {similar['curriculo_codigo']} | SIMILARIDADE: {similar['similaridade']:.1%}
   
   DISCIPLINA: {similar['curriculo_eixo']}
   
   HABILIDADE: {similar['curriculo_objetivo']}
   
   ORIENTAÇÕES PEDAGÓGICAS: {similar['curriculo_exemplos']}
   
   {'─' * 50}
"""
    
    return relatorio_txt

def gerar_relatorio_csv():
    dados_csv = []
    
    for habilidade in relatorio_completo:
        # Com a busca adaptativa, todas as habilidades terão pelo menos uma correspondência
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
                'Nota_Corte_Usada': f"{habilidade['nota_corte_usada']:.1%}",
                'Busca_Adaptativa': 'Não' if habilidade['tem_similaridade_original'] else 'Sim',
                'Data_Analise': estatisticas['data_analise']
            })
    
    return pd.DataFrame(dados_csv)

def gerar_resumo_executivo():
    resumo = f"""
==================================================================================
                            RESUMO EXECUTIVO
==================================================================================
Data: {estatisticas['data_analise']}
Nota de corte inicial: {estatisticas['nota_corte_original']*100}%
Busca adaptativa: ATIVADA
Modelo: {estatisticas['modelo_usado']}

PRINCIPAIS DESCOBERTAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• {estatisticas['bncc_com_similaridade_original']} de {estatisticas['total_bncc']} habilidades da BNCC ({estatisticas['bncc_com_similaridade_original']/estatisticas['total_bncc']*100:.1f}%) têm correspondência com nota de corte original ≥ {estatisticas['nota_corte_original']*100}%

• {estatisticas['bncc_com_similaridade_adaptativa']} de {estatisticas['total_bncc']} habilidades da BNCC ({estatisticas['bncc_com_similaridade_adaptativa']/estatisticas['total_bncc']*100:.1f}%) têm correspondência usando busca adaptativa

• Total de {estatisticas['total_matches_acima_corte']} conexões identificadas acima da nota de corte original

• Nota de corte média usada: {np.mean(estatisticas['notas_corte_usadas']):.1%}
• Nota de corte mínima usada: {np.min(estatisticas['notas_corte_usadas']):.1%}

HABILIDADES BNCC COM MAIOR NÚMERO DE CORRESPONDÊNCIAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    top_correspondencias = sorted(relatorio_completo, key=lambda x: x['quantidade_similares'], reverse=True)[:CONFIGURACOES['MOSTRAR_TOP_CORRESPONDENCIAS']]
    
    for i, hab in enumerate(top_correspondencias, 1):
        resumo += f"\n{i:2d}. {hab['bncc_codigo']} - {hab['quantidade_similares']} correspondências (máx: {hab['maior_similaridade']:.1%})"
        resumo += f"\n    {hab['bncc_eixo']}"
        resumo += f"\n    Nota de corte usada: {hab['nota_corte_usada']:.1%} {'(original)' if hab['tem_similaridade_original'] else '(adaptativa)'}"
    
    resumo += f"\n\nHABILIDADES QUE PRECISARAM DE BUSCA ADAPTATIVA:\n"
    resumo += "━" * 70 + "\n"
    
    busca_adaptativa = [h for h in relatorio_completo if not h['tem_similaridade_original']]
    for hab in busca_adaptativa:
        resumo += f"\n• {hab['bncc_codigo']} - {hab['bncc_eixo']}"
        resumo += f"\n  Nota de corte usada: {hab['nota_corte_usada']:.1%}"
        resumo += f"\n  Máx. similaridade: {hab['maior_similaridade']:.1%}"
        resumo += f"\n  {hab['bncc_objetivo'][:100]}{'...' if len(hab['bncc_objetivo']) > 100 else ''}\n"
    
    return resumo

# ==================================================================================
#                           SALVAMENTO DOS ARQUIVOS
# ==================================================================================

# Criar pasta docs/anos finais se não existir
docs_path = os.path.join("docs", "anos finais")
if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    print(f"📁 Pasta '{docs_path}' criada para organizar os relatórios")

nome_relatorio_completo = os.path.join(docs_path, f"anosfinais_{CONFIGURACOES['PREFIXO_ARQUIVOS']}relatorio_completo.txt")
nome_relatorio_csv = os.path.join(docs_path, f"anosfinais_{CONFIGURACOES['PREFIXO_ARQUIVOS']}relatorio.csv")
nome_resumo = os.path.join(docs_path, f"anosfinais_{CONFIGURACOES['PREFIXO_ARQUIVOS']}resumo_executivo.txt")
nome_heatmap = os.path.join(docs_path, f"anosfinais_{CONFIGURACOES['PREFIXO_ARQUIVOS']}heatmap_similaridade.png")

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
        # Extrai códigos para usar como índices (usando as colunas corretas)
        bncc_codigos = bncc_df_inf['HABILIDADE'].apply(extrair_codigo).tolist()
        curriculo_codigos = curriculo_df_inf['HABILIDADES'].apply(extrair_codigo).tolist()
        
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
print(f"⚙️  Nota de corte inicial: {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"📊 {estatisticas['bncc_com_similaridade_original']}/{estatisticas['total_bncc']} habilidades BNCC têm correspondência ≥ {CONFIGURACOES['NOTA_CORTE']*100}%")
print(f"� {estatisticas['bncc_com_similaridade_adaptativa']}/{estatisticas['total_bncc']} habilidades BNCC têm correspondência com busca adaptativa")
print(f"�🔍 {estatisticas['total_matches_acima_corte']} conexões identificadas acima da nota de corte original")
print(f"📈 Nota de corte média usada: {np.mean(estatisticas['notas_corte_usadas']):.1%}")
print(f"📉 Nota de corte mínima usada: {np.min(estatisticas['notas_corte_usadas']):.1%}")

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
print("💡 DICA: Para alterar a nota de corte inicial, modifique CONFIGURACOES['NOTA_CORTE'] no topo do script!")
print("🔧 A busca adaptativa garante que toda habilidade BNCC tenha pelo menos uma correspondência!")
print("📁 Consulte os arquivos de relatório para análise completa!")
print("="*80)