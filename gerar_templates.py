#!/usr/bin/env python3
"""
Script para gerar templates de currículo em Excel para cada segmento educacional
"""

import pandas as pd
import os

def criar_template_infantil():
    """Cria o template para Educação Infantil"""
    print("🎨 Criando template para Educação Infantil...")
    
    # Dados de exemplo para orientar o usuário
    dados_exemplo = [
        {
            'SEGMENTO': 'EDUCAÇÃO INFANTIL',
            'ANO': '1º PERÍODO',
            'EIXO': 'CORPO, GESTOS E MOVIMENTOS',
            'OBJETIVO DE APRENDIZAGEM': '(EI03CG01) Criar com o corpo formas diversificadas de expressão de sentimentos, sensações e emoções, tanto nas situações do cotidiano quanto em brincadeiras, dança, teatro e música.',
            'EXEMPLOS': 'Representar-se em situações de brincadeiras ou teatro, apresentando suas características corporais, seus interesses, sentimentos, sensações ou emoções.'
        },
        {
            'SEGMENTO': 'EDUCAÇÃO INFANTIL',
            'ANO': '2º PERÍODO',
            'EIXO': 'ESCUTA, FALA, PENSAMENTO E IMAGINAÇÃO',
            'OBJETIVO DE APRENDIZAGEM': '(EI03EF01) Expressar ideias, desejos e sentimentos sobre suas vivências, por meio da linguagem oral e escrita.',
            'EXEMPLOS': 'Comunicar-se com diferentes finalidades: perguntar, informar, relatar, etc.'
        },
        # Linha vazia para o usuário preencher
        {
            'SEGMENTO': '',
            'ANO': '',
            'EIXO': '',
            'OBJETIVO DE APRENDIZAGEM': '',
            'EXEMPLOS': ''
        }
    ]
    
    df = pd.DataFrame(dados_exemplo)
    caminho = os.path.join('templates_curriculo', 'template_educacao_infantil.xlsx')
    
    with pd.ExcelWriter(caminho, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Currículo Infantil', index=False)
        
        # Adicionar uma aba de instruções
        instrucoes = pd.DataFrame([
            ['INSTRUÇÕES PARA PREENCHIMENTO'],
            [''],
            ['1. SEGMENTO: Sempre "EDUCAÇÃO INFANTIL"'],
            ['2. ANO: 1º PERÍODO, 2º PERÍODO, BERÇÁRIO I, BERÇÁRIO II, MATERNAL, etc.'],
            ['3. EIXO: Campos de experiência da BNCC:'],
            ['   - O EU, O OUTRO E O NÓS'],
            ['   - CORPO, GESTOS E MOVIMENTOS'],
            ['   - TRAÇOS, SONS, CORES E FORMAS'],
            ['   - ESCUTA, FALA, PENSAMENTO E IMAGINAÇÃO'],
            ['   - ESPAÇOS, TEMPOS, QUANTIDADES, RELAÇÕES E TRANSFORMAÇÕES'],
            ['4. OBJETIVO DE APRENDIZAGEM: Código e descrição da habilidade (ex: (EI03CG01) Descrição...)'],
            ['5. EXEMPLOS: Exemplos práticos de como trabalhar a habilidade'],
            [''],
            ['IMPORTANTE:'],
            ['- Delete as linhas de exemplo antes de enviar'],
            ['- Preencha apenas com dados do seu currículo municipal'],
            ['- Use os códigos da BNCC quando possível'],
            ['- Seja específico nos exemplos práticos']
        ])
        instrucoes.to_excel(writer, sheet_name='INSTRUÇÕES', index=False, header=False)
    
    print(f"✅ Template criado: {caminho}")
    return caminho

def criar_template_anos_iniciais():
    """Cria o template para Anos Iniciais"""
    print("📚 Criando template para Anos Iniciais...")
    
    # Dados de exemplo
    dados_exemplo = [
        {
            'SEGMENTO': 'ANOS INICIAIS',
            'ANO': '1º Ano',
            'DISCIPLINA': 'ARTE',
            'HABILIDADES': '(EF15AR01) Identificar e apreciar formas distintas das artes visuais tradicionais e contemporâneas, cultivando a percepção, o imaginário, a capacidade de simbolizar e o repertório imagético.',
            'ORIENTACOES_PEDAGOGICAS': 'Proporcionar experiências diversificadas em artes visuais, cultivando a percepção, o imaginário e a capacidade de simbolizar dos alunos.'
        },
        {
            'SEGMENTO': 'ANOS INICIAIS',
            'ANO': '2º Ano',
            'DISCIPLINA': 'PORTUGUÊS',
            'HABILIDADES': '(EF02LP01) Utilizar, ao produzir o texto, grafia correta de palavras conhecidas ou com estruturas silábicas já dominadas, letras maiúsculas em início de frases e em substantivos próprios, segmentação entre as palavras, ponto final, ponto de interrogação e ponto de exclamação.',
            'ORIENTACOES_PEDAGOGICAS': 'Trabalhar sistematicamente a grafia de palavras conhecidas e estruturas silábicas através de atividades práticas e contextualizadas.'
        },
        # Linha vazia para o usuário
        {
            'SEGMENTO': '',
            'ANO': '',
            'DISCIPLINA': '',
            'HABILIDADES': '',
            'ORIENTACOES_PEDAGOGICAS': ''
        }
    ]
    
    df = pd.DataFrame(dados_exemplo)
    caminho = os.path.join('templates_curriculo', 'template_anos_iniciais.xlsx')
    
    with pd.ExcelWriter(caminho, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Currículo Anos Iniciais', index=False)
        
        # Aba de instruções
        instrucoes = pd.DataFrame([
            ['INSTRUÇÕES PARA PREENCHIMENTO'],
            [''],
            ['1. SEGMENTO: Sempre "ANOS INICIAIS"'],
            ['2. ANO: 1º Ano, 2º Ano, 3º Ano, 4º Ano, 5º Ano'],
            ['3. DISCIPLINA: Disciplinas do currículo:'],
            ['   - ARTE'],
            ['   - CIÊNCIAS'],
            ['   - EDUCAÇÃO FÍSICA'],
            ['   - GEOGRAFIA'],
            ['   - HISTÓRIA'],
            ['   - PORTUGUÊS'],
            ['   - MATEMÁTICA'],
            ['   - ENSINO RELIGIOSO'],
            ['4. HABILIDADES: Código da BNCC e descrição (ex: (EF01LP01) Descrição...)'],
            ['5. ORIENTAÇÕES PEDAGÓGICAS: Como trabalhar a habilidade na prática'],
            [''],
            ['DICAS:'],
            ['- Use os códigos oficiais da BNCC'],
            ['- Seja específico nas orientações pedagógicas'],
            ['- Delete as linhas de exemplo antes de enviar'],
            ['- Uma linha por habilidade']
        ])
        instrucoes.to_excel(writer, sheet_name='INSTRUÇÕES', index=False, header=False)
    
    print(f"✅ Template criado: {caminho}")
    return caminho

def criar_template_anos_finais():
    """Cria o template para Anos Finais"""
    print("🎓 Criando template para Anos Finais...")
    
    # Dados de exemplo
    dados_exemplo = [
        {
            'SEGMENTO': 'ANOS FINAIS',
            'ANO': '6º Ano',
            'DISCIPLINA': 'HISTÓRIA',
            'HABILIDADES': '(EF06HI01) Identificar diferentes formas de compreensão da noção de tempo e de periodização dos processos históricos (continuidades e rupturas).',
            'ORIENTACOES_PEDAGOGICAS': 'Conceituar e identificar as diferentes noções de tempo (cronológico, climático e histórico). Conhecer as variadas formas de notação do tempo.'
        },
        {
            'SEGMENTO': 'ANOS FINAIS',
            'ANO': '7º Ano',
            'DISCIPLINA': 'MATEMÁTICA',
            'HABILIDADES': '(EF07MA01) Resolver e elaborar problemas com números inteiros e racionais, envolvendo as operações fundamentais.',
            'ORIENTACOES_PEDAGOGICAS': 'Trabalhar com situações-problema do cotidiano que envolvam números inteiros e racionais, explorando diferentes estratégias de resolução.'
        },
        # Linha vazia para o usuário
        {
            'SEGMENTO': '',
            'ANO': '',
            'DISCIPLINA': '',
            'HABILIDADES': '',
            'ORIENTACOES_PEDAGOGICAS': ''
        }
    ]
    
    df = pd.DataFrame(dados_exemplo)
    caminho = os.path.join('templates_curriculo', 'template_anos_finais.xlsx')
    
    with pd.ExcelWriter(caminho, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Currículo Anos Finais', index=False)
        
        # Aba de instruções
        instrucoes = pd.DataFrame([
            ['INSTRUÇÕES PARA PREENCHIMENTO'],
            [''],
            ['1. SEGMENTO: Sempre "ANOS FINAIS"'],
            ['2. ANO: 6º Ano, 7º Ano, 8º Ano, 9º Ano'],
            ['3. DISCIPLINA: Disciplinas do currículo:'],
            ['   - ARTE'],
            ['   - CIÊNCIAS'],
            ['   - EDUCAÇÃO FÍSICA'],
            ['   - GEOGRAFIA'],
            ['   - HISTÓRIA'],
            ['   - PORTUGUÊS'],
            ['   - MATEMÁTICA'],
            ['   - INGLÊS'],
            ['   - ENSINO RELIGIOSO'],
            ['4. HABILIDADES: Código da BNCC e descrição (ex: (EF06HI01) Descrição...)'],
            ['5. ORIENTAÇÕES PEDAGÓGICAS: Metodologias e estratégias pedagógicas'],
            [''],
            ['OBSERVAÇÕES:'],
            ['- Mantenha a estrutura das colunas'],
            ['- Use os códigos oficiais da BNCC quando disponíveis'],
            ['- Seja detalhado nas orientações pedagógicas'],
            ['- Delete as linhas de exemplo antes de enviar'],
            ['- Verifique se todas as células estão preenchidas']
        ])
        instrucoes.to_excel(writer, sheet_name='INSTRUÇÕES', index=False, header=False)
    
    print(f"✅ Template criado: {caminho}")
    return caminho

if __name__ == "__main__":
    print("="*60)
    print("🏗️  GERADOR DE TEMPLATES DE CURRÍCULO")
    print("="*60)
    
    # Criar diretório se não existir
    if not os.path.exists('templates_curriculo'):
        os.makedirs('templates_curriculo')
    
    templates_criados = []
    
    # Criar todos os templates
    templates_criados.append(criar_template_infantil())
    templates_criados.append(criar_template_anos_iniciais())
    templates_criados.append(criar_template_anos_finais())
    
    print("\n" + "="*60)
    print("✅ TEMPLATES CRIADOS COM SUCESSO!")
    print("="*60)
    
    for template in templates_criados:
        print(f"📁 {template}")
    
    print("\n💡 Os templates incluem:")
    print("   📋 Exemplos de preenchimento")
    print("   📖 Instruções detalhadas")
    print("   ✅ Estrutura correta das colunas")
    print("   🎯 Orientações específicas por segmento")
