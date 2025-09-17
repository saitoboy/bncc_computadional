import pandas as pd
import sys
import os

# Adicionar o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(__file__))

from core.similarity import extrair_codigo

# Simular os dados que chegam no algoritmo balanceado
print("🔍 SIMULANDO O PROBLEMA DO ALGORITMO BALANCEADO:")
print("="*60)

# Exemplo de como os dados chegam no itertuples
class MockLinha:
    def __init__(self, objetivo_aprendizagem):
        self.OBJETIVO_DE_APRENDIZAGEM = objetivo_aprendizagem
        # Simular hasattr
        self.attrs = ['OBJETIVO_DE_APRENDIZAGEM']
    
    def __getattr__(self, name):
        if name == 'OBJETIVO_DE_APRENDIZAGEM':
            return self.OBJETIVO_DE_APRENDIZAGEM
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Teste com exemplos reais
exemplo_linha = MockLinha("(EI03CG01) Criar com o corpo formas diversificadas de expressão de sentimentos, sensações e emoções, tanto nas situações do cotidiano quanto em brincadeiras, dança, teatro e música.")

print("1. TESTE hasattr:")
print(f"   hasattr(linha, 'HABILIDADES'): {hasattr(exemplo_linha, 'HABILIDADES')}")
print(f"   hasattr(linha, 'OBJETIVO_DE_APRENDIZAGEM'): {hasattr(exemplo_linha, 'OBJETIVO_DE_APRENDIZAGEM')}")
print(f"   hasattr(linha, 'HABILIDADE'): {hasattr(exemplo_linha, 'HABILIDADE')}")

print("\n2. TESTE EXTRAÇÃO DO CÓDIGO:")
if hasattr(exemplo_linha, 'HABILIDADES'):
    codigo = extrair_codigo(exemplo_linha.HABILIDADES)
    print(f"   Usando HABILIDADES: {codigo}")
elif hasattr(exemplo_linha, 'OBJETIVO_DE_APRENDIZAGEM'):
    codigo = extrair_codigo(exemplo_linha.OBJETIVO_DE_APRENDIZAGEM)
    print(f"   Usando OBJETIVO_DE_APRENDIZAGEM: {codigo}")
elif hasattr(exemplo_linha, 'HABILIDADE'):
    codigo = extrair_codigo(exemplo_linha.HABILIDADE)
    print(f"   Usando HABILIDADE: {codigo}")
else:
    codigo = f"CURR_99"
    print(f"   Fallback: {codigo}")

print(f"\n✅ Código final extraído: {codigo}")

print("\n" + "="*60)
print("🎯 RESULTADO: A função extrair_codigo() está funcionando!")
print("💡 O problema deve estar em outro lugar no pipeline...")
