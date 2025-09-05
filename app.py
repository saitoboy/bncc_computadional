from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from core.similarity import process_uploaded_file

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'troque_para_uma_chave_secreta'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Validar arquivo
    if 'file' not in request.files:
        flash('Nenhum arquivo enviado')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('Nenhum arquivo selecionado')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Formato não permitido. Envie .xlsx, .xls ou .csv')
        return redirect(url_for('index'))

    segment = request.form.get('segment')
    nota_corte = float(request.form.get('nota_corte', 0.8))

    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(saved_path)

    # Processar (sincrono)
    try:
        print(f"🔄 Processando arquivo: {filename}")
        print(f"📋 Segmento: {segment}")
        print(f"🎯 Nota de corte: {nota_corte}")
        
        resultado = process_uploaded_file(saved_path, segment, nota_corte)
        
        print("✅ Processamento concluído com sucesso!")
        
        # Armazenar informações da última análise para as rotas de relatório
        app.config['LAST_ANALYSIS'] = resultado
        
        return render_template('results.html', 
                             resumo=resultado.get('resumo'), 
                             files=resultado.get('files'), 
                             top_matches=resultado.get('top_matches'),
                             segment=segment,
                             nota_corte=nota_corte)
                             
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erro durante o processamento: {error_msg}")
        
        # Mensagens de erro mais amigáveis
        if "Colunas ausentes" in error_msg:
            flash(f"Estrutura de arquivo incorreta: {error_msg}")
        elif "não encontrado" in error_msg:
            flash(f"Arquivo de referência não encontrado: {error_msg}")
        elif "Formato de arquivo" in error_msg:
            flash("Formato de arquivo não suportado. Use .xlsx, .xls ou .csv")
        else:
            flash(f"Erro durante o processamento: {error_msg}")
        
        return redirect(url_for('index'))

@app.route('/download/<path:filepath>')
def download(filepath):
    # filepath é relativo à pasta src
    full_path = os.path.join(BASE_DIR, filepath)
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/download_template/<template_name>')
def download_template(template_name):
    """Download dos templates de currículo"""
    templates = {
        'infantil': 'template_educacao_infantil.xlsx',
        'anos_iniciais': 'template_anos_iniciais.xlsx',
        'anos_finais': 'template_anos_finais.xlsx'
    }
    
    if template_name not in templates:
        flash('Template não encontrado')
        return redirect(url_for('index'))
    
    template_path = os.path.join(BASE_DIR, 'templates_curriculo')
    filename = templates[template_name]
    
    return send_from_directory(template_path, filename, as_attachment=True)

@app.route('/get_report/<report_type>')
def get_report(report_type):
    """Serve os relatórios detalhados para as abas"""
    try:
        # Usar os dados da última análise
        last_analysis = app.config.get('LAST_ANALYSIS')
        if not last_analysis:
            return "Nenhuma análise recente encontrada. Faça uma nova análise primeiro."
        
        files = last_analysis.get('files', {})
        
        if report_type == 'executive':
            # Buscar arquivo de resumo executivo
            resumo_path = files.get('resumo_executivo')
            if resumo_path:
                full_path = os.path.join(BASE_DIR, resumo_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        return f.read()
            return "Resumo executivo não encontrado."
                
        elif report_type == 'detailed':
            # Buscar arquivo de relatório detalhado
            detalhado_path = files.get('relatorio_detalhado')
            if detalhado_path:
                full_path = os.path.join(BASE_DIR, detalhado_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        return f.read()
            return "Relatório detalhado não encontrado."
        
        return "Tipo de relatório não reconhecido."
        
    except Exception as e:
        return f"Erro ao carregar relatório: {e}"

if __name__ == '__main__':
    app.run(debug=True)
