import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from transcript import transcribe_audio
from structured import process_file, save_structured_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRANSCRIPT_FOLDER'] = 'transcripts'
app.config['STRUCTURED_FOLDER'] = 'structured'
app.config['SECRET_KEY'] = 'your-secret-key-here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STRUCTURED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_files_info(folder):
    files = []
    for filename in os.listdir(folder):
        if not filename.endswith('.html'):
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                files.append({
                    'name': filename,
                    'size': os.path.getsize(path),
                    'upload_time': os.path.getctime(path)
                })
    files.sort(key=lambda x: x['upload_time'], reverse=True)
    return files

def read_structured_file(filename):
    filepath = os.path.join(app.config['STRUCTURED_FOLDER'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return None

@app.route('/view/<filename>')
def view_structured_file(filename):
    content = read_structured_file(filename)
    if content:
        return render_template('index.html',
                            audio_files=get_files_info(app.config['UPLOAD_FOLDER']),
                            transcript_files=get_files_info(app.config['TRANSCRIPT_FOLDER']),
                            structured_files=get_files_info(app.config['STRUCTURED_FOLDER']),
                            info_text=content,
                            current_file=filename[:-4]+'txt')
    flash('Файл не найден или не может быть прочитан')
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    audio_files = get_files_info(app.config['UPLOAD_FOLDER'])
    transcript_files = get_files_info(app.config['TRANSCRIPT_FOLDER'])
    structured_files = get_files_info(app.config['STRUCTURED_FOLDER'])

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Файл не выбран')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('Файл успешно загружен')
            return redirect(url_for('index'))
        else:
            flash('Недопустимый тип файла. Разрешены только MP3.')

    return render_template('index.html',
                         audio_files=audio_files,
                         transcript_files=transcript_files,
                         structured_files=structured_files,
                         info_text="Выберите файл для просмотра его содержимого.")

@app.route('/transcribe/<filename>')
def transcribe_file(filename):
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], output_filename)

        transcribed_text = transcribe_audio(input_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)

        flash('Файл успешно пересказан')
    except Exception as e:
        flash(f'Ошибка при пересказе файла: {str(e)}')

    return redirect(url_for('index'))

@app.route('/process/<filename>')
def process_text_file(filename):
    try:
        input_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], filename)
        output_base = os.path.splitext(filename)[0]
        output_base_path = os.path.join(app.config['STRUCTURED_FOLDER'], output_base)


        max_workers = min(8, (os.cpu_count() or 1) * 2)

        print(f"Используется {max_workers} рабочих потоков...")
        result = process_file(input_path, max_workers=max_workers)
        save_structured_text(result, output_base_path)


        content = read_structured_file(output_base + '.html')
        flash('Файл успешно обработан')
        return render_template('index.html',
                            audio_files=get_files_info(app.config['UPLOAD_FOLDER']),
                            transcript_files=get_files_info(app.config['TRANSCRIPT_FOLDER']),
                            structured_files=get_structured_files(),
                            info_text=content,
                            current_file=output_base + '.html')
    except Exception as e:
        flash(f'Ошибка при обработке файла: {str(e)}')
        return redirect(url_for('index'))

def get_structured_files():
    files = []
    for filename in os.listdir(app.config['STRUCTURED_FOLDER']):
        if filename.endswith('.txt'):
            path = os.path.join(app.config['STRUCTURED_FOLDER'], filename)
            if os.path.isfile(path):
                files.append({
                    'name': filename,
                    'size': os.path.getsize(path),
                    'upload_time': os.path.getctime(path)
                })
    files.sort(key=lambda x: x['upload_time'], reverse=True)
    return files

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    if folder == 'transcripts':
        directory = app.config['TRANSCRIPT_FOLDER']
    elif folder == 'structured':
        directory = app.config['STRUCTURED_FOLDER']
    else:
        flash('Неверный путь для скачивания')
        return redirect(url_for('index'))

    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        return send_from_directory(directory, filename, as_attachment=True)

    flash('Файл не найден')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
