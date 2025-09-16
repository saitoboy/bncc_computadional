# 🎯 Algoritmo Balanceado de Correspondência BNCC x Currículo

## 📋 Problema Identificado

O algoritmo anterior concentrava as correspondências em uma única disciplina, resultando em:
- **Habilidades duplicadas** sendo utilizadas múltiplas vezes
- **Concentração excessiva** em disciplinas com vocabulário similar
- **Falta de representação equilibrada** entre as diferentes áreas do conhecimento
- **Dificuldade de análise crítica** pelos educadores

## ✅ Solução Implementada

### 🎲 Algoritmo Balanceado por Disciplinas

O novo algoritmo `encontrar_similaridade_balanceada()` implementa uma estratégia multi-camadas:

#### 🏗️ **Etapa 1: Agrupamento por Disciplina**
```python
disciplinas_curriculo = {
    'Matemática': [hab1, hab2, hab3...],
    'Português': [hab4, hab5, hab6...],
    'Ciências': [hab7, hab8, hab9...],
    # ... outras disciplinas
}
```

#### 🎯 **Etapa 2: Estratégias de Correspondência**

**ESTRATÉGIA 1 - Nota de Corte Original**
- Busca a **melhor habilidade de cada disciplina** que atenda à nota de corte inicial (80%)
- Garante **distribuição equilibrada** entre disciplinas
- **Prioriza qualidade** das correspondências

**ESTRATÉGIA 2 - Busca Adaptativa**
- Se não encontrou correspondências com nota original, **reduz gradualmente** (1% por vez)
- Ainda mantém o **princípio de uma habilidade por disciplina**
- Garante **pelo menos uma correspondência** por habilidade BNCC

**ESTRATÉGIA 3 - Fallback Garantido**
- Como último recurso, pega a **melhor correspondência geral disponível**
- Evita que qualquer habilidade BNCC fique **sem correspondência**

**ESTRATÉGIA 4 - Enriquecimento Controlado**
- Adiciona até **3 correspondências por habilidade BNCC**
- Apenas se a similaridade for ≥ **90% da nota de corte usada**
- Mantém **qualidade alta** das correspondências adicionais

#### 🚫 **Controle de Duplicatas**
```python
habilidades_ja_usadas = set()  # Rastreia códigos únicos
```
- Cada código de habilidade do currículo é usado **no máximo uma vez**
- Evita **redundância** no mapeamento
- Garante **uso eficiente** do currículo municipal

## 📊 Benefícios Alcançados

### 🎯 **Distribuição Equilibrada**
- **Múltiplas disciplinas** representadas nas correspondências
- **Evita concentração** em uma única área do conhecimento
- **Reflete a interdisciplinaridade** da educação

### 🚫 **Eliminação de Duplicatas**
- **Códigos únicos**: Cada habilidade do currículo aparece no máximo uma vez
- **Aproveitamento eficiente** do currículo municipal
- **Relatórios mais limpos** e organizados

### 📈 **Qualidade Mantida**
- **Prioriza correspondências** com maior similaridade
- **Nota de corte adaptativa** quando necessário
- **Transparência total** sobre as decisões do algoritmo

### 🔍 **Análise Crítica Facilitada**
- Educadores podem **avaliar cada correspondência** individualmente
- **Suprimir habilidades** que consideram inadequadas
- **Refinar o mapeamento** baseado na experiência pedagógica

## 📋 Saídas do Relatório

### 📊 **Estatísticas Adicionais**
```
📊 ESTATÍSTICAS DE DISTRIBUIÇÃO:
   🎯 Habilidades do currículo utilizadas: 156/200 (78.0%)
   🚫 Habilidades não utilizadas: 44
   📚 Distribuição de uso por disciplina:
      Matemática: 45/60 (75.0%)
      Português: 38/50 (76.0%)
      Ciências: 28/40 (70.0%)
      História: 25/30 (83.3%)
      Geografia: 20/20 (100.0%)
```

### 🔍 **Verificação de Qualidade**
```
🔍 VERIFICAÇÃO FINAL DOS CÓDIGOS EXTRAÍDOS:
BNCC 1: (EF01MA01)
Objetivo: Utilizar números naturais como indicador de quantidade...
Nota de corte usada: 80.0%
Similaridades encontradas: 2
Disciplinas envolvidas: 2
Melhor match: (CURR_MAT_15) (85.2%)
Disciplina: Matemática
```

### 📈 **Resumo Executivo Enriquecido**
- **Benefícios do algoritmo balanceado** explicados
- **Distribuição por disciplinas** detalhada
- **Habilidades únicas utilizadas** vs. disponíveis
- **Recomendações** para análise crítica

## 🔧 Implementação Técnica

### 📁 **Arquivos Modificados**
- `anosiniciais.py` - Implementação completa do algoritmo balanceado
- `anosfinais.py` - Mesma implementação para anos finais
- Função `encontrar_similaridade_balanceada()` - Novo algoritmo principal
- Função `encontrar_similaridade_adaptativa()` - Mantida para compatibilidade

### ⚙️ **Configuração**
```python
CONFIGURACOES = {
    'NOTA_CORTE': 0.80,  # Nota de corte inicial
    'MAX_CORRESPONDENCIAS_POR_BNCC': 3,  # Máximo de correspondências por habilidade
    'PERCENTUAL_ENRIQUECIMENTO': 0.9,  # 90% da nota de corte para correspondências adicionais
}
```

## 🚀 Próximos Passos

### 🔄 **Integração com o App Flask**
- Atualizar `core/similarity.py` para usar o novo algoritmo
- Manter **compatibilidade** com os três segmentos educacionais
- Preservar todas as **funcionalidades existentes**

### 📊 **Melhorias Futuras**
- **Interface web** para análise crítica das correspondências
- **Filtros por disciplina** nos relatórios
- **Exportação personalizada** baseada nas escolhas dos educadores
- **Histórico de refinamentos** aplicados

### 🎯 **Validação Pedagógica**
- **Testes com educadores** para validar a qualidade das correspondências
- **Ajustes nos pesos** por disciplina se necessário
- **Feedback loop** para melhoria contínua do algoritmo

---

**Autor**: Guilherme Saito  
**Data**: 16 de setembro de 2025  
**Versão**: 2.0 - Algoritmo Balanceado por Disciplinas
