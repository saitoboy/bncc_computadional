# 📊 Comparador BNCC x Currículo Municipal

## 📝 Descrição

Este aplicativo web desenvolvido em Flask permite comparar as habilidades da **Base Nacional Comum Curricular (BNCC)** com currículos municipais de educação, utilizando análise de similaridade semântica com inteligência artificial.

## 🎯 Objetivo

Facilitar o trabalho de educadores e gestores educacionais na identificação de correspondências entre as competências definidas pela BNCC e as habilidades presentes nos currículos municipais, garantindo alinhamento pedagógico e cumprimento das diretrizes nacionais.

## 🔧 Funcionalidades Principais

### 📥 Upload de Arquivos
- **Formatos suportados**: Excel (.xlsx, .xls) e CSV
- **Drag & Drop**: Interface intuitiva para envio de arquivos
- **Validação automática**: Verificação de formato e estrutura dos dados

### 🎓 Segmentos Educacionais
O sistema suporta três segmentos da educação básica:

1. **Educação Infantil**
2. **Anos Iniciais do Ensino Fundamental**  
3. **Anos Finais do Ensino Fundamental**

### 📋 Templates para Download
- **Templates pré-formatados** para cada segmento educacional
- **Estrutura padronizada** com colunas obrigatórias
- **Facilita** o preenchimento correto dos dados

### 🤖 Análise de Similaridade
- **Modelo de IA**: Utiliza `SentenceTransformer` para análise semântica
- **Busca adaptativa**: Garante pelo menos uma correspondência para cada habilidade BNCC
- **Nota de corte configurável**: Padrão de 80% de similaridade
- **Proxy corporativo**: Configurado para ambientes empresariais

### 📊 Relatórios Completos

#### 📈 Visualizações
- **Heatmap de similaridade** com rótulos informativos
- **Gráficos interativos** mostrando correspondências
- **Interface web responsiva** com abas organizadas

#### 📄 Relatórios Textuais
- **Resumo Executivo**: Principais descobertas e estatísticas
- **Relatório Completo**: Análise detalhada de cada habilidade
- **Dados em CSV**: Para análises adicionais

#### 🔍 Métricas Detalhadas
- Total de habilidades analisadas
- Percentual de correspondências encontradas
- Estatísticas da busca adaptativa
- Top correspondências por similaridade

### 💾 Downloads
- **CSV completo** com todas as correspondências
- **Heatmap em PNG** alta resolução
- **Relatórios em TXT** formatados

## 🏗️ Arquitetura Técnica

### 📁 Estrutura do Projeto
```
src/
├── app.py                      # Aplicação Flask principal
├── core/
│   └── similarity.py          # Módulo de análise de similaridade
├── templates/
│   ├── index.html            # Página de upload
│   └── results.html          # Página de resultados
├── templates_excel/          # Templates para download
│   ├── template_infantil.xlsx
│   ├── template_anos_iniciais.xlsx
│   └── template_anos_finais.xlsx
├── uploads/                  # Arquivos enviados pelos usuários
├── docs/                    # Relatórios gerados
└── bncc_*.xlsx             # Arquivos base da BNCC
```

### 🔄 Fluxo de Processamento

1. **Upload**: Usuário envia arquivo do currículo municipal
2. **Validação**: Sistema verifica estrutura e colunas obrigatórias
3. **Carregamento**: Dados da BNCC correspondente ao segmento são carregados
4. **Análise**: Algoritmo de IA calcula similaridade semântica
5. **Busca Adaptativa**: Garante correspondência para todas as habilidades
6. **Geração**: Cria relatórios, gráficos e arquivos de download
7. **Apresentação**: Exibe resultados em interface web organizada

### 🧠 Tecnologias Utilizadas

#### Backend
- **Flask**: Framework web Python
- **Pandas**: Manipulação de dados
- **SentenceTransformers**: Modelo de IA para embeddings
- **Scikit-learn**: Cálculo de similaridade cosseno
- **OpenPyXL**: Leitura/escrita de Excel

#### Frontend
- **HTML5**: Estrutura das páginas
- **CSS3**: Estilização responsiva
- **JavaScript**: Interatividade (drag & drop)
- **Bootstrap**: Framework CSS (parcial)

#### Visualização
- **Matplotlib**: Geração de gráficos
- **Seaborn**: Heatmaps estatísticos
- **Plotly** (futuro): Gráficos interativos

### 📊 Estruturas de Dados

#### Educação Infantil
**BNCC**: `EIXO`, `OBJETIVO DE APRENDIZAGEM`, `EXEMPLOS`  
**Currículo**: `EIXO`, `OBJETIVO DE APRENDIZAGEM`, `EXEMPLOS`, `ANO`

#### Anos Iniciais/Finais
**BNCC**: `EIXO`, `HABILIDADE`, `EXEMPLOS`  
**Currículo**: `DISCIPLINA`, `HABILIDADES`, `ORIENTACOES_PEDAGOGICAS`

## 🚀 Como Usar

### 1. Preparação dos Dados
- Baixe o template correspondente ao seu segmento
- Preencha as colunas obrigatórias com dados do currículo municipal
- Salve o arquivo em formato Excel ou CSV

### 2. Upload e Análise
- Acesse a página inicial do sistema
- Arraste seu arquivo para a área de upload
- Selecione o segmento educacional
- Ajuste a nota de corte (opcional)
- Clique em "Analisar Similaridade"

### 3. Visualização de Resultados
- **Aba Resumo**: Principais estatísticas e descobertas
- **Aba Heatmap**: Mapa visual de similaridades
- **Aba Top Matches**: Melhores correspondências encontradas
- **Aba Relatório Executivo**: Resumo gerencial
- **Aba Relatório Completo**: Análise detalhada

### 4. Download de Resultados
- CSV com dados completos para análise externa
- Heatmap em alta resolução para apresentações
- Relatórios formatados para documentação

## ⚙️ Configurações Técnicas

### Busca Adaptativa
- **Nota inicial**: 80% de similaridade
- **Redução gradual**: 1% por iteração se necessário
- **Garantia**: Pelo menos uma correspondência por habilidade BNCC
- **Transparência**: Indica quando foi usada busca adaptativa

### Limites e Validações
- **Tamanho máximo**: Configurável via Flask
- **Formatos aceitos**: .xlsx, .xls, .csv
- **Validação de colunas**: Automática por segmento
- **Tratamento de erros**: Mensagens claras ao usuário

## 📈 Benefícios

### Para Educadores
- **Identificação rápida** de gaps curriculares
- **Mapeamento preciso** entre BNCC e currículo local
- **Evidências quantitativas** para tomada de decisão
- **Relatórios prontos** para apresentação

### Para Gestores
- **Conformidade** com diretrizes nacionais
- **Análise comparativa** entre diferentes currículos
- **Dados objetivos** para planejamento pedagógico
- **Documentação completa** para auditorias

### Para Redes de Ensino
- **Padronização** de processos de análise
- **Escalabilidade** para múltiplas escolas
- **Histórico** de comparações realizadas
- **Interface amigável** reduzindo curva de aprendizado

## 🔮 Futuras Melhorias

- **Autenticação de usuários** para controle de acesso
- **Histórico de análises** por usuário/escola
- **Comparação entre currículos** de diferentes municípios
- **Exportação para Word/PDF** dos relatórios
- **API REST** para integração com outros sistemas
- **Dashboard analítico** com métricas avançadas
- **Notificações** de conclusão de processamento
- **Modo offline** para ambientes sem internet

## 👨‍💻 Desenvolvimento

**Autor**: Guilherme Saito  
**Tecnologia**: Python Flask + IA  
**Licença**: [Definir]  
**Repositório**: [URL do repositório]

---

*Este documento foi gerado automaticamente baseado na análise do código-fonte do sistema.*