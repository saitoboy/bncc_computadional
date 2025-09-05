#!/usr/bin/env python3
"""
Script de teste para verificar se o processamento funciona corretamente
"""

import os
import sys
from core.similarity import process_uploaded_file

def test_infantil():
    """Testa o processamento para educação infantil"""
    print("🔄 Testando segmento INFANTIL...")
    
    # Arquivo de teste
    test_file = os.path.join("uploads", "curriculo_df_inf.xlsx")
    
    if not os.path.exists(test_file):
        print(f"❌ Arquivo de teste não encontrado: {test_file}")
        return False
    
    try:
        resultado = process_uploaded_file(test_file, "infantil", 0.8)
        print("✅ Teste INFANTIL passou!")
        print(f"📊 Resumo: {len(resultado['resumo'])} chaves")
        print(f"📁 Arquivos: {len(resultado['files'])} arquivos")
        print(f"🏆 Top matches: {len(resultado['top_matches'])} matches")
        return True
    except Exception as e:
        print(f"❌ Teste INFANTIL falhou: {e}")
        return False

def test_anos_iniciais():
    """Testa o processamento para anos iniciais"""
    print("\n🔄 Testando segmento ANOS INICIAIS...")
    
    # Arquivo de teste
    test_file = os.path.join("uploads", "ANOSINICIAIS_curriculo_normalizado.xlsx")
    
    if not os.path.exists(test_file):
        print(f"❌ Arquivo de teste não encontrado: {test_file}")
        return False
    
    try:
        resultado = process_uploaded_file(test_file, "anos iniciais", 0.8)
        print("✅ Teste ANOS INICIAIS passou!")
        print(f"📊 Resumo: {len(resultado['resumo'])} chaves")
        print(f"📁 Arquivos: {len(resultado['files'])} arquivos")
        print(f"🏆 Top matches: {len(resultado['top_matches'])} matches")
        return True
    except Exception as e:
        print(f"❌ Teste ANOS INICIAIS falhou: {e}")
        return False

def test_anos_finais():
    """Testa o processamento para anos finais"""
    print("\n🔄 Testando segmento ANOS FINAIS...")
    
    # Arquivo de teste
    test_file = os.path.join("uploads", "Anos_Finais_Curriculo_Normalizado.xlsx")
    
    if not os.path.exists(test_file):
        print(f"❌ Arquivo de teste não encontrado: {test_file}")
        return False
    
    try:
        resultado = process_uploaded_file(test_file, "anos finais", 0.8)
        print("✅ Teste ANOS FINAIS passou!")
        print(f"📊 Resumo: {len(resultado['resumo'])} chaves")
        print(f"📁 Arquivos: {len(resultado['files'])} arquivos")
        print(f"🏆 Top matches: {len(resultado['top_matches'])} matches")
        return True
    except Exception as e:
        print(f"❌ Teste ANOS FINAIS falhou: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTANDO PROCESSAMENTO DE SIMILARIDADE")
    print("="*60)
    
    # Mudar para o diretório correto
    os.chdir(os.path.dirname(__file__))
    
    testes_passou = []
    
    # Testar cada segmento
    testes_passou.append(test_infantil())
    testes_passou.append(test_anos_iniciais())
    testes_passou.append(test_anos_finais())
    
    print("\n" + "="*60)
    print("📊 RESULTADOS DOS TESTES")
    print("="*60)
    
    total_testes = len(testes_passou)
    testes_ok = sum(testes_passou)
    
    print(f"✅ Testes aprovados: {testes_ok}/{total_testes}")
    
    if testes_ok == total_testes:
        print("🎉 Todos os testes passaram! O sistema está funcionando.")
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
        sys.exit(1)
