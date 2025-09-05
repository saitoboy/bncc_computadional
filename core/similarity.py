import os
from dotenv import load_dotenv
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

# Configurar proxy para Hugging Face
proxy_config = {
    'http': 'http://guilherme.saito:890484gS@10.10.30.9:3128',
    'https': 'http://guilherme.saito:890484gS@10.10.30.9:3128'
}

os.environ['HTTP_PROXY'] = proxy_config['http']
os.environ['HTTPS_PROXY'] = proxy_config['https']
os.environ['http_proxy'] = proxy_config['http']
os.environ['https_proxy'] = proxy_config['https']

# Contornar problemas de SSL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Funções refatoradas para uso pelo Flask

def extrair_codigo(obj):
    if pd.isna(obj):
        return "(SEM_COD)"
    obj_str = str(obj)
    patterns = [
        r'\(([A-Z]{2}\d{2}[A-Z]{2}\d{2})\)',
        r'\(([A-Z]+\d+[A-Z]*\d*)\)',
        r'\(([A-Z0-9]+)\)'
    ]
    for pattern in patterns:
        match = re.search(pattern, obj_str)
        if match:
            return f"({match.group(1)})"
    match_inicio = re.search(r'^\(([^)]+)\)', obj_str)
    if match_inicio:
        return f"({match_inicio.group(1)})"
    return f"({obj_str[:15]}...)" if len(obj_str) > 15 else f"({obj_str})"


def concat_features_bncc(df):
    """
    Concatena features da BNCC, adaptando-se às diferentes estruturas de colunas
    """
    # Para infantil: usa 'OBJETIVO DE APRENDIZAGEM'
    # Para outros segmentos: usa 'HABILIDADE'
    if 'OBJETIVO DE APRENDIZAGEM' in df.columns:
        # Estrutura do infantil
        return (
            df['EIXO'].astype(str) + " | " +
            df['OBJETIVO DE APRENDIZAGEM'].astype(str) + " | " +
            df['EXEMPLOS'].astype(str)
        )
    else:
        # Estrutura dos anos iniciais e finais
        return (
            df['EIXO'].astype(str) + " | " +
            df['HABILIDADE'].astype(str) + " | " +
            df['EXEMPLOS'].astype(str)
        )


def concat_features_curriculo(df):
    """
    Concatena features do currículo municipal, adaptando-se às diferentes estruturas
    """
    # Para infantil: usa 'OBJETIVO DE APRENDIZAGEM'
    # Para outros segmentos: usa 'HABILIDADES' ou 'HABILIDADE'
    if 'OBJETIVO DE APRENDIZAGEM' in df.columns:
        # Estrutura do infantil
        return (
            df['EIXO'].astype(str) + " | " +
            df['OBJETIVO DE APRENDIZAGEM'].astype(str) + " | " +
            df['EXEMPLOS'].astype(str)
        )
    else:
        # Estrutura dos anos iniciais e finais - detectar qual coluna de habilidade existe
        habilidade_col = 'HABILIDADES' if 'HABILIDADES' in df.columns else 'HABILIDADE'
        return (
            df['DISCIPLINA'].astype(str) + " | " +
            df[habilidade_col].astype(str) + " | " +
            df['ORIENTACOES_PEDAGOGICAS'].astype(str)
        )


def encontrar_similaridade_adaptativa(similaridades_bncc, nota_corte_inicial):
    nota_corte_atual = nota_corte_inicial
    indices_similares = np.where(similaridades_bncc >= nota_corte_atual)[0]
    while len(indices_similares) == 0 and nota_corte_atual > 0.1:
        nota_corte_atual -= 0.01
        indices_similares = np.where(similaridades_bncc >= nota_corte_atual)[0]
    if len(indices_similares) == 0:
        idx_melhor = np.argmax(similaridades_bncc)
        indices_similares = np.array([idx_melhor])
        nota_corte_atual = similaridades_bncc[idx_melhor]
    return indices_similares, nota_corte_atual


# Processar o arquivo enviado pelo usuário
def process_uploaded_file(uploaded_path, segment, nota_corte):
    # Ler o arquivo do usuário
    ext = os.path.splitext(uploaded_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        user_df = pd.read_excel(uploaded_path)
    elif ext == '.csv':
        user_df = pd.read_csv(uploaded_path)
    else:
        raise Exception('Formato de arquivo não suportado')

    # Decidir qual BNCC usar com base no segmento
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Volta para src/
    
    if 'infantil' in segment.lower():
        # Para infantil, usar sempre o arquivo BNCC específico do infantil
        bncc_path = os.path.join(base_dir, 'bncc_df_inf.xlsx')
        # Se não existir, tentar o arquivo normalizado
        if not os.path.exists(bncc_path):
            bncc_path = os.path.join(base_dir, 'data', 'infantil_curriculo', 'infantil_curriculo_normalizado.xlsx')
        curriculo_df = user_df
    elif 'iniciais' in segment.lower():
        bncc_path = os.path.join(base_dir, 'bncc_df_anosiniciais.xlsx')
        curriculo_df = user_df
    elif 'finais' in segment.lower():
        bncc_path = os.path.join(base_dir, 'bncc_df_anosfinais (2).xlsx')
        curriculo_df = user_df
    else:
        raise Exception(f'Segmento inválido: {segment}')

    # Verificar se arquivos existem
    if not os.path.exists(bncc_path):
        raise Exception(f'Arquivo BNCC não encontrado: {bncc_path}')
    
    print(f"📁 Carregando BNCC de: {bncc_path}")
    print(f"📁 Currículo do usuário: {uploaded_path}")
    
    # Carregar BNCC
    bncc_df = pd.read_excel(bncc_path)
    
    print(f"📊 BNCC carregada: {len(bncc_df)} linhas")
    print(f"📊 Currículo carregado: {len(curriculo_df)} linhas")
    print(f"🔍 Colunas BNCC: {list(bncc_df.columns)}")
    print(f"🔍 Colunas Currículo: {list(curriculo_df.columns)}")

    # Normalizar colunas
    bncc_df.columns = bncc_df.columns.str.strip()
    curriculo_df.columns = curriculo_df.columns.str.strip()

    # Validação flexível de colunas baseada no segmento
    def validate_columns(df, df_type, segment):
        """Valida se as colunas necessárias estão presentes"""
        cols = set(df.columns)
        
        if 'infantil' in segment.lower():
            if df_type == 'bncc':
                required = ['EIXO', 'OBJETIVO DE APRENDIZAGEM', 'EXEMPLOS']
            else:  # curriculo
                # Para infantil, o currículo pode ter ANO ou não
                required = ['EIXO', 'OBJETIVO DE APRENDIZAGEM', 'EXEMPLOS']
                # ANO é opcional para infantil
        else:
            if df_type == 'bncc':
                required = ['EIXO', 'HABILIDADE', 'EXEMPLOS']
            else:  # curriculo
                # Aceitar variações comuns nas colunas do currículo
                if 'HABILIDADES' in cols:
                    required = ['DISCIPLINA', 'HABILIDADES', 'ORIENTACOES_PEDAGOGICAS']
                elif 'HABILIDADE' in cols:
                    required = ['DISCIPLINA', 'HABILIDADE', 'ORIENTACOES_PEDAGOGICAS']
                else:
                    required = ['DISCIPLINA', 'HABILIDADES', 'ORIENTACOES_PEDAGOGICAS']
        
        missing = [col for col in required if col not in cols]
        return missing, required
    
    # Validar BNCC
    missing_bncc, required_bncc = validate_columns(bncc_df, 'bncc', segment)
    if missing_bncc:
        raise Exception(f'Colunas ausentes no arquivo BNCC: {missing_bncc}. Colunas necessárias: {required_bncc}. Colunas disponíveis: {list(bncc_df.columns)}')
    
    # Validar currículo
    missing_curriculo, required_curriculo = validate_columns(curriculo_df, 'curriculo', segment)
    if missing_curriculo:
        raise Exception(f'Colunas ausentes no arquivo do currículo: {missing_curriculo}. Colunas necessárias: {required_curriculo}. Colunas disponíveis: {list(curriculo_df.columns)}')

    # Gerar textos
    bncc_texts = concat_features_bncc(bncc_df)
    curriculo_texts = concat_features_curriculo(curriculo_df)

    # Carregar modelo
    print("Carregando modelo de embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embeddings
    print("Gerando embeddings...")
    bncc_embeddings = model.encode(bncc_texts.tolist(), show_progress_bar=False)
    curriculo_embeddings = model.encode(curriculo_texts.tolist(), show_progress_bar=False)

    grau_similaridade = cosine_similarity(bncc_embeddings, curriculo_embeddings)

    # Gerar relatório simples
    output_dir = os.path.join(base_dir, 'docs', segment.replace(' ', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    relatorio = []
    notas_usadas = []

    for idx, row in bncc_df.iterrows():
        sims = grau_similaridade[idx]
        indices, nota_usada = encontrar_similaridade_adaptativa(sims, nota_corte)
        notas_usadas.append(nota_usada)
        melhores = sorted([(i, sims[i]) for i in indices], key=lambda x: x[1], reverse=True)
        
        # Adaptar para diferentes estruturas de colunas
        if 'infantil' in segment.lower():
            bncc_habilidade_col = 'OBJETIVO DE APRENDIZAGEM'
            curriculo_habilidade_col = 'OBJETIVO DE APRENDIZAGEM'
        else:
            bncc_habilidade_col = 'HABILIDADE'
            # Detectar qual coluna de habilidade existe no currículo
            curriculo_habilidade_col = 'HABILIDADES' if 'HABILIDADES' in curriculo_df.columns else 'HABILIDADE'
        
        for i, score in melhores:
            relatorio.append({
                'bncc_indice': idx+1,
                'bncc_codigo': extrair_codigo(row[bncc_habilidade_col]),
                'bncc_objetivo': row[bncc_habilidade_col],
                'curriculo_indice': i+1,
                'curriculo_codigo': extrair_codigo(curriculo_df.iloc[i][curriculo_habilidade_col]),
                'similaridade': score
            })

    # Salvar CSV
    df_out = pd.DataFrame(relatorio)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{segment.replace(' ', '_')}_relatorio_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Gerar relatórios de texto (resumo executivo e detalhado)
    resumo_executivo = gerar_resumo_executivo(relatorio, bncc_df, curriculo_df, notas_usadas, nota_corte, segment, timestamp)
    relatorio_detalhado = gerar_relatorio_detalhado(relatorio, bncc_df, curriculo_df, notas_usadas, nota_corte, segment, timestamp)
    
    # Salvar relatórios de texto
    resumo_filename = f"{segment.replace(' ', '_')}_resumo_executivo_{timestamp}.txt"
    resumo_path = os.path.join(output_dir, resumo_filename)
    with open(resumo_path, 'w', encoding='utf-8') as f:
        f.write(resumo_executivo)
    
    detalhado_filename = f"{segment.replace(' ', '_')}_relatorio_completo_{timestamp}.txt"
    detalhado_path = os.path.join(output_dir, detalhado_filename)
    with open(detalhado_path, 'w', encoding='utf-8') as f:
        f.write(relatorio_detalhado)

    # Gerar heatmap
    heatmap_path = None
    try:
        # Definir colunas baseado no segmento
        if 'infantil' in segment.lower():
            bncc_col = 'OBJETIVO DE APRENDIZAGEM'
            curriculo_col = 'OBJETIVO DE APRENDIZAGEM'
        else:
            bncc_col = 'HABILIDADE'
            # Detectar qual coluna de habilidade existe no currículo
            curriculo_col = 'HABILIDADES' if 'HABILIDADES' in curriculo_df.columns else 'HABILIDADE'
        
        # Extrair códigos mais limpos para os rótulos
        bncc_codigos = []
        for idx, row in bncc_df.iterrows():
            codigo = extrair_codigo(row[bncc_col])
            # Limitar o tamanho para melhor visualização
            if len(codigo) > 12:
                codigo = codigo[:10] + "..."
            bncc_codigos.append(codigo)
        
        curr_codigos = []
        for idx, row in curriculo_df.iterrows():
            codigo = extrair_codigo(row[curriculo_col])
            # Limitar o tamanho para melhor visualização
            if len(codigo) > 12:
                codigo = codigo[:10] + "..."
            curr_codigos.append(codigo)
        
        # Criar DataFrame de similaridade
        sim_df = pd.DataFrame(grau_similaridade, index=bncc_codigos, columns=curr_codigos)
        
        # Configurar tamanho baseado na quantidade de dados
        max_size = 25  # Aumentar um pouco mais para melhor visualização
        rows_to_show = min(max_size, len(sim_df))
        cols_to_show = min(max_size, len(sim_df.columns))
        
        # Configurar matplotlib para melhor qualidade
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Criar heatmap com configurações melhoradas
        heatmap = sns.heatmap(
            sim_df.iloc[:rows_to_show, :cols_to_show], 
            cmap='Blues', 
            vmin=0, 
            vmax=1,
            annot=True,  # Mostrar valores
            fmt='.2f',   # Formato dos valores
            cbar_kws={
                'label': 'Similaridade Semântica',
                'shrink': 0.8
            },
            square=True,  # Células quadradas
            linewidths=0.5,  # Linhas entre células
            linecolor='white',
            ax=ax
        )
        
        # Configurar título e labels
        ax.set_title(
            f'Mapa de Calor - Similaridade Semântica\n{segment.title()} (BNCC × Currículo Municipal)', 
            fontsize=16, 
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Habilidades do Currículo Municipal', fontsize=12, fontweight='bold')
        ax.set_ylabel('Habilidades da BNCC', fontsize=12, fontweight='bold')
        
        # Melhorar a rotação e tamanho dos rótulos
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # Adicionar informações adicionais
        info_text = f"""
Modelo: all-MiniLM-L6-v2 | Nota de corte: {nota_corte:.0%}
Exibindo {rows_to_show} × {cols_to_show} primeiras habilidades
Cores mais escuras = maior similaridade semântica
        """.strip()
        
        plt.figtext(0.02, 0.02, info_text, fontsize=9, style='italic', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar com alta qualidade
        heatmap_filename = f"{segment.replace(' ', '_')}_heatmap_{timestamp}.png"
        heatmap_path = os.path.join(output_dir, heatmap_filename)
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Heatmap salvo: {heatmap_path}")
        
    except Exception as e:
        print(f"Erro ao gerar heatmap: {e}")
        heatmap_path = None

    resumo = {
        'total_bncc': len(bncc_df),
        'total_curriculo': len(curriculo_df),
        'nota_media_usada': float(np.mean(notas_usadas)),
        'csv': os.path.relpath(csv_path, base_dir),
        'heatmap': os.path.relpath(heatmap_path, base_dir) if heatmap_path else None,
        'resumo_executivo': os.path.relpath(resumo_path, base_dir),
        'relatorio_detalhado': os.path.relpath(detalhado_path, base_dir),
        'modelo_usado': 'all-MiniLM-L6-v2',
        'matches_acima_80': sum(1 for item in relatorio if item['similaridade'] >= 0.8)
    }

    # Arquivos para download
    files = {
        'csv': resumo['csv'], 
        'heatmap': resumo['heatmap'],
        'resumo_executivo': resumo['resumo_executivo'],
        'relatorio_detalhado': resumo['relatorio_detalhado']
    }

    top_matches = df_out.sort_values('similaridade', ascending=False).head(10).to_dict(orient='records')

    return {
        'resumo': resumo, 
        'files': files, 
        'top_matches': top_matches
    }

def gerar_resumo_executivo(relatorio, bncc_df, curriculo_df, notas_usadas, nota_corte, segment, timestamp):
    """Gera o resumo executivo da análise"""
    estatisticas = {
        'total_bncc': len(bncc_df),
        'total_curriculo': len(curriculo_df),
        'nota_corte_original': nota_corte,
        'nota_media_usada': np.mean(notas_usadas),
        'nota_min_usada': np.min(notas_usadas),
        'data_analise': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Contar habilidades com nota original
    bncc_com_nota_original = sum(1 for n in notas_usadas if n >= nota_corte)
    
    # Agrupar por habilidade BNCC para contar correspondências
    from collections import defaultdict
    correspondencias_por_bncc = defaultdict(list)
    for item in relatorio:
        correspondencias_por_bncc[item['bncc_codigo']].append(item)
    
    # Top habilidades com mais correspondências
    top_correspondencias = sorted(
        correspondencias_por_bncc.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )[:15]
    
    resumo = f"""
==================================================================================
                            RESUMO EXECUTIVO
==================================================================================
Data: {estatisticas['data_analise']}
Nota de corte inicial: {estatisticas['nota_corte_original']*100:.1f}%
Busca adaptativa: ATIVADA
Modelo: all-MiniLM-L6-v2

PRINCIPAIS DESCOBERTAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• {bncc_com_nota_original} de {estatisticas['total_bncc']} habilidades da BNCC ({bncc_com_nota_original/estatisticas['total_bncc']*100:.1f}%) têm correspondência com nota de corte original ≥ {estatisticas['nota_corte_original']*100:.1f}%

• {estatisticas['total_bncc']} de {estatisticas['total_bncc']} habilidades da BNCC (100.0%) têm correspondência usando busca adaptativa

• Nota de corte média usada: {estatisticas['nota_media_usada']:.1%}
• Nota de corte mínima usada: {estatisticas['nota_min_usada']:.1%}

HABILIDADES BNCC COM MAIOR NÚMERO DE CORRESPONDÊNCIAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for i, (codigo, correspondencias) in enumerate(top_correspondencias, 1):
        max_sim = max(c['similaridade'] for c in correspondencias)
        primeiro_item = correspondencias[0]
        resumo += f"\n{i:2d}. {codigo} - {len(correspondencias)} correspondências (máx: {max_sim:.1%})"
        resumo += f"\n    Nota de corte usada: {primeiro_item.get('nota_corte_usada', 'N/A')}"
    
    return resumo


def gerar_relatorio_detalhado(relatorio, bncc_df, curriculo_df, notas_usadas, nota_corte, segment, timestamp):
    """Gera o relatório detalhado da análise"""
    estatisticas = {
        'total_bncc': len(bncc_df),
        'total_curriculo': len(curriculo_df),
        'nota_corte_original': nota_corte,
        'nota_media_usada': np.mean(notas_usadas),
        'nota_min_usada': np.min(notas_usadas),
        'data_analise': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    bncc_com_nota_original = sum(1 for n in notas_usadas if n >= nota_corte)
    
    relatorio_txt = f"""
==================================================================================
                    RELATÓRIO DE SIMILARIDADE BNCC x CURRÍCULO MUNICIPAL
==================================================================================
Data do relatório: {estatisticas['data_analise']}
Nota de corte inicial: {estatisticas['nota_corte_original']*100:.1f}% de similaridade
Modelo utilizado: all-MiniLM-L6-v2
Busca adaptativa: ATIVADA (garante pelo menos 1 correspondência por habilidade)

ESTATÍSTICAS GERAIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total de habilidades BNCC analisadas: {estatisticas['total_bncc']}
• Habilidades BNCC com similaridade ≥ {estatisticas['nota_corte_original']*100:.1f}% (nota original): {bncc_com_nota_original} ({bncc_com_nota_original/estatisticas['total_bncc']*100:.1f}%)
• Habilidades BNCC com correspondência (busca adaptativa): {estatisticas['total_bncc']} (100.0%)
• Nota de corte média usada: {estatisticas['nota_media_usada']:.1%}
• Nota de corte mínima usada: {estatisticas['nota_min_usada']:.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELATÓRIO DETALHADO POR CORRESPONDÊNCIA:

"""
    
    # Agrupar por habilidade BNCC
    from collections import defaultdict
    correspondencias_por_bncc = defaultdict(list)
    for item in relatorio:
        correspondencias_por_bncc[item['bncc_codigo']].append(item)
    
    for i, (codigo, correspondencias) in enumerate(correspondencias_por_bncc.items(), 1):
        primeiro = correspondencias[0]
        relatorio_txt += f"""
================================================================================
HABILIDADE BNCC #{primeiro['bncc_indice']} - {codigo}
================================================================================

OBJETIVO BNCC: {primeiro['bncc_objetivo']}

CORRESPONDÊNCIAS ENCONTRADAS: {len(correspondencias)}

HABILIDADES SIMILARES DO CURRÍCULO:
------------------------------------------------------------"""
        
        for j, corr in enumerate(correspondencias, 1):
            relatorio_txt += f"""
{j}. CURRÍCULO #{corr['curriculo_indice']} - {corr['curriculo_codigo']} | SIMILARIDADE: {corr['similaridade']:.1%}
"""
    
    return relatorio_txt
