from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import os
import traceback
from werkzeug.utils import secure_filename
from core.similarity import process_uploaded_file
from datetime import datetime

# Configuração inicial
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'templates')
DOCS_FOLDER = os.path.join(BASE_DIR, 'docs')

# Criar pastas necessárias
for folder in [UPLOAD_FOLDER, TEMPLATES_FOLDER, DOCS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

app = Flask(__name__, template_folder=TEMPLATES_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.secret_key = 'chave_secreta_producao_2026'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Página principal com formulário de upload"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Processa arquivo e gera análise BNCC x Currículo"""
    
    # Validações básicas
    if 'file' not in request.files:
        flash('❌ Nenhum arquivo enviado', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('❌ Nenhum arquivo selecionado', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('❌ Formato inválido. Use .xlsx, .xls ou .csv', 'error')
        return redirect(url_for('index'))

    # Parâmetros
    segment = request.form.get('segment', '').strip()
    try:
        nota_corte = float(request.form.get('nota_corte', '0.8'))
        if not 0.1 <= nota_corte <= 1.0:
            raise ValueError("entre 10% e 100%")
    except ValueError:
        flash('❌ Nota de corte deve ser um número entre 0.1 e 1.0', 'error')
        return redirect(url_for('index'))

    if not segment:
        flash('❌ Selecione um segmento educacional', 'error')
        return redirect(url_for('index'))

    # Salvar arquivo
    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(saved_path)

    print(f"\n🚀 INICIANDO ANÁLISE BNCC x CURRÍCULO")
    print(f"📁 Arquivo: {filename}")
    print(f"📋 Segmento: {segment}")
    print(f"🎯 Nota de corte: {nota_corte:.1%}")

    try:
        # Processar arquivo
        resultado_raw = process_uploaded_file(saved_path, segment, nota_corte)
        
        # Garantir estrutura válida do resultado
        resultado = _normalizar_resultado(resultado_raw, segment, nota_corte, 
                                        filename.replace('.xlsx', ''))
        
        # Salvar na sessão
        app.config['LAST_ANALYSIS'] = {
            'resultado': resultado,
            'segment': segment,
            'nota_corte': nota_corte,
            'timestamp': datetime.now().isoformat(),
            'arquivo': filename
        }
        
        print(f"✅ ANÁLISE CONCLUÍDA! Renderizando results.html")
        return render_template('results.html',
                             resumo=resultado['resumo'],
                             files=resultado['files'],
                             top_matches=resultado['top_matches'],
                             segment=segment,
                             nota_corte=nota_corte)

    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERRO: {error_msg}")
        print(f"📋 TRACEBACK:\n{traceback.format_exc()}")
        
        flash(_mensagem_erro_usuario(error_msg), 'error')
        return redirect(url_for('index'))

def _normalizar_resultado(resultado_raw, segment, nota_corte, nome_arquivo):
    """Garante que o resultado sempre tenha a estrutura esperada"""
    
    if resultado_raw is None:
        resultado_raw = {}
    
    if not isinstance(resultado_raw, dict):
        resultado_raw = {}
    
    # Resumo padrão
    resumo = resultado_raw.get('resumo', {})
    resumo_base = {
        'total_bncc': resumo.get('total_bncc', 0),
        'total_curriculo': resumo.get('total_curriculo', 0),
        'habilidades_utilizadas': resumo.get('habilidades_utilizadas', 0),
        'disciplinas_envolvidas': resumo.get('disciplinas_envolvidas', 0),
        'nota_media_usada': nota_corte,
        'eficiencia_uso': 0,
        'algoritmo': 'Balanceado por Disciplinas ✅',
        'segmento': segment,
        'status': 'completo',
        'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M')
    }
    
    # Arquivos (fallback vazio)
    files = resultado_raw.get('files', {})
    files_base = {
        'csv': files.get('csv'),
        'heatmap': files.get('heatmap'), 
        'resumo_executivo': files.get('resumo_executivo'),
        'relatorio_detalhado': files.get('relatorio_detalhado')
    }
    
    # Top matches (fallback vazio)
    top_matches_raw = resultado_raw.get('top_matches', [])
    top_matches = []
    for match in top_matches_raw[:10]:
        if isinstance(match, dict):
            top_matches.append({
                'bncc_codigo': match.get('bncc_codigo', 'N/A'),
                'bncc_objetivo': match.get('bncc_objetivo', ''),
                'bncc_indice': match.get('bncc_indice', 0),
                'curriculo_codigo': match.get('curriculo_codigo', 'N/A'),
                'curriculo_objetivo': match.get('curriculo_objetivo', ''),
                'curriculo_indice': match.get('curriculo_indice', 0),
                'similaridade': match.get('similaridade', 0),
                'nota_corte_usada': match.get('nota_corte_usada', nota_corte)
            })
    
    return {
        'resumo': {**resumo_base, **resumo},
        'files': files_base,
        'top_matches': top_matches
    }

def _mensagem_erro_usuario(error_msg):
    """Mensagens de erro amigáveis para o usuário"""
    if "Colunas ausentes" in error_msg or "colunas" in error_msg.lower():
        return "❌ Arquivo deve ter colunas: SEGMENTO, EIXO, OBJETIVO DE APRENDIZAGEM, EXEMPLOS"
    elif "não encontrado" in error_msg.lower():
        return "❌ Arquivo BNCC de referência não encontrado no projeto"
    elif "modelo" in error_msg.lower():
        return "❌ Erro no modelo de IA. Tente novamente em alguns minutos"
    elif "chaves" in error_msg.lower():
        return "❌ Erro interno do sistema. Contate o suporte técnico"
    else:
        return f"❌ Erro inesperado: {error_msg[:80]}..."

@app.route('/download/<path:filepath>')
def download(filepath):
    """Download de arquivos gerados"""
    try:
        full_path = os.path.join(BASE_DIR, filepath)
        if not os.path.exists(full_path):
            flash('Arquivo não encontrado')
            return redirect(url_for('index'))
        
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception:
        flash('Erro ao baixar arquivo')
        return redirect(url_for('index'))

@app.route('/download_template/<template_name>')
def download_template(template_name):
    """Download dos templates de exemplo"""
    templates = {
        'infantil': 'template_educacao_infantil.xlsx',
        'anos_iniciais': 'template_anos_iniciais.xlsx', 
        'anos_finais': 'template_anos_finais.xlsx'
    }
    
    if template_name not in templates:
        flash('Template não encontrado')
        return redirect(url_for('index'))
    
    template_path = os.path.join(BASE_DIR, 'templates_curriculo')
    if not os.path.exists(template_path):
        flash('Pasta de templates não encontrada')
        return redirect(url_for('index'))
    
    filename = templates[template_name]
    full_path = os.path.join(template_path, filename)
    
    if not os.path.exists(full_path):
        flash(f'Template {filename} não encontrado')
        return redirect(url_for('index'))
    
    return send_from_directory(template_path, filename, as_attachment=True)

@app.route('/get_report/<report_type>')
def get_report(report_type):
    """API para carregar relatórios nas abas do results.html"""
    try:
        last_analysis = app.config.get('LAST_ANALYSIS', {})
        files = last_analysis.get('resultado', {}).get('files', {})
        
        if report_type == 'executive':
            path = files.get('resumo_executivo')
            if path and os.path.exists(os.path.join(BASE_DIR, path)):
                with open(os.path.join(BASE_DIR, path), 'r', encoding='utf-8') as f:
                    return f.read()
            return "📋 RESUMO EXECUTIVO\n\nNenhum relatório executivo encontrado.\nFaça uma nova análise para gerar os relatórios detalhados."
        
        elif report_type == 'detailed':
            path = files.get('relatorio_detalhado')
            if path and os.path.exists(os.path.join(BASE_DIR, path)):
                with open(os.path.join(BASE_DIR, path), 'r', encoding='utf-8') as f:
                    return f.read()
            return "📄 RELATÓRIO DETALHADO\n\nNenhum relatório detalhado encontrado.\nFaça uma nova análise para gerar os relatórios."
        
        return "Tipo de relatório inválido", 400
        
    except Exception as e:
        return f"Erro ao carregar relatório: {str(e)}", 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return "Erro interno do servidor", 500

if __name__ == '__main__':
    print("🚀 Servidor BNCC x Currículo iniciado!")
    print(f"📁 Pasta uploads: {UPLOAD_FOLDER}")
    print(f"📁 Pasta templates: {TEMPLATES_FOLDER}")
    app.run(debug=True, host='127.0.0.1', port=5000)
