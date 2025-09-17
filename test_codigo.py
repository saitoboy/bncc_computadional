import pandas as pd
import sys
import os

# Adicionar o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(__file__))

from core.similarity import extrair_codigo

# Teste com exemplos reais do infantil
exemplos_infantil = [
    "(EI03CG01) Criar com o corpo formas diversificadas de expressão de sentimentos, sensações e emoções, tanto nas situações do cotidiano quanto em brincadeiras, dança, teatro e música.",
    "(EI03CG02) Demonstrar controle e adequação do uso de seu corpo em brincadeiras e jogos, escuta e reconto de histórias, atividades artísticas, entre outras possibilidades.",
    "(EI03CG03) Criar movimentos, gestos, olhares e mímicas em brincadeiras, jogos e atividades artísticas como dança, teatro e música.",
    "(EI01EF23ME03) Reconhecer os combinados e rotinas por meio de gravuras.",
    "(EI03EF02) Inventar brincadeiras cantadas, poemas e canções, criando rimas, alterações e ritmos.",
]

print("🔍 TESTE DA FUNÇÃO extrair_codigo():")
print("="*50)

for i, exemplo in enumerate(exemplos_infantil, 1):
    codigo_extraido = extrair_codigo(exemplo)
    print(f"{i}. Texto: {exemplo[:50]}...")
    print(f"   Código extraído: {codigo_extraido}")
    print()

print("="*50)
print("✅ Teste concluído!")
