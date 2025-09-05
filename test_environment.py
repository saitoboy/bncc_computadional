#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.dirname(__file__))

def test_structure():
    """Verifica se a estrutura de arquivos está correta"""
    print("=== VERIFICAÇÃO DA ESTRUTURA ===")
    
    required_files = [
        'app.py',
        'core/similarity.py',
        'templates/index.html',
        'templates/results.html'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - ARQUIVO AUSENTE")
    
    # Verificar pastas necessárias
    required_dirs = [
        'uploads',
        'docs'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ - SERÁ CRIADO AUTOMATICAMENTE")

def test_imports():
    """Testa se todas as importações estão funcionando"""
    print("\n=== VERIFICAÇÃO DAS IMPORTAÇÕES ===")
    
    try:
        from flask import Flask
        print("✅ Flask importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Flask: {e}")
        return False
    
    try:
        from core.similarity import process_uploaded_file
        print("✅ Módulo similarity importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar similarity: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar pandas: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar SentenceTransformers: {e}")
        return False
    
    return True

def check_data_files():
    """Verifica se os arquivos de dados da BNCC estão presentes"""
    print("\n=== VERIFICAÇÃO DOS ARQUIVOS DE DADOS ===")
    
    data_files = [
        'bncc_df_anosiniciais.xlsx',
        'bncc_df_anosfinais (2).xlsx',
        'data/infantil_curriculo/infantil_curriculo_normalizado.xlsx'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - ARQUIVO AUSENTE")

def main():
    print("🚀 TESTE DO AMBIENTE FLASK - COMPARADOR BNCC")
    print("=" * 50)
    
    # Mudar para o diretório src se necessário
    if not os.path.exists('app.py'):
        print("❌ app.py não encontrado no diretório atual")
        print("💡 Certifique-se de estar no diretório src/")
        return
    
    test_structure()
    
    if not test_imports():
        print("\n❌ FALHA NAS IMPORTAÇÕES - Instale as dependências necessárias")
        print("💡 Execute: pip install flask pandas sentence-transformers scikit-learn matplotlib seaborn openpyxl")
        return
    
    check_data_files()
    
    print("\n" + "=" * 50)
    print("✅ AMBIENTE CONFIGURADO COM SUCESSO!")
    print("\n🌐 Para iniciar o servidor Flask, execute:")
    print("   python app.py")
    print("\n📝 O servidor estará disponível em: http://localhost:5000")
    print("\n💡 Dicas:")
    print("   - Arraste arquivos .xlsx/.csv com colunas: DISCIPLINA, HABILIDADES, ORIENTACOES_PEDAGOGICAS")
    print("   - Selecione o segmento correto (infantil, anos iniciais, anos finais)")
    print("   - Ajuste a nota de corte conforme necessário")

if __name__ == "__main__":
    main()
