import whisper
import os

def transcribe_audio(input_path: str, language: str="ru") -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл {input_path} не найден!")

    model = whisper.load_model('base')

    result = model.transcribe(input_path, language=language, verbose=True, fp16=False)

    return result["text"]
