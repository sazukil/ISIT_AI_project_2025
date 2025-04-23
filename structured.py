from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import Counter
import re
import spacy
from spacy.lang.ru.stop_words import STOP_WORDS
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

model_lock = threading.Lock()

class ModelStorage:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._spacy_nlp = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            with model_lock:
                model_name = "IlyaGusev/rut5_base_sum_gazeta"
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            with model_lock:
                model_name = "IlyaGusev/rut5_base_sum_gazeta"
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return self._model

    @property
    def spacy_nlp(self):
        if self._spacy_nlp is None:
            with model_lock:
                self._spacy_nlp = spacy.load("ru_core_news_sm")
        return self._spacy_nlp

_thread_local = threading.local()

def get_models():
    if not hasattr(_thread_local, "model_storage"):
        _thread_local.model_storage = ModelStorage()
    return _thread_local.model_storage

def extract_keywords_spacy(text: str, top_n: int = 15) -> list:
    models = get_models()
    doc = models.spacy_nlp(text)

    keywords = [
        token.lemma_ for token in doc
        if token.lemma_ not in STOP_WORDS
        and not token.is_punct
        and token.pos_ in {"NOUN", "PROPN", "ADJ"}
    ]

    counter = Counter(keywords)
    return counter.most_common(top_n)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
PUNCTUATION_CLEAN_RE = re.compile(r"[^\w\s]")

def split_text_by_sentences(text: str, sentences_per_chunk: int = 6) -> list:
    sentences = SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]

def summarize_text(text: str) -> str:
    models = get_models()
    inputs = models.tokenizer([text], max_length=1000, truncation=True, return_tensors="pt")
    summary_ids = models.model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=150,
        min_length=50,
        no_repeat_ngram_size=3,
        early_stopping=True,
        length_penalty=1.0
    )
    return models.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def parallel_summarize_chunks(chunks: list, max_workers: int = None) -> list:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(summarize_text, chunk) for chunk in chunks]
        return [future.result() for future in tqdm(as_completed(futures), total=len(chunks), desc="Суммаризация")]

def parallel_extract_keywords(text_chunks: list, top_n: int = 15, max_workers: int = None) -> list:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_keywords_spacy, chunk, top_n) for chunk in text_chunks]
        keyword_counts = []

        for future in tqdm(as_completed(futures), total=len(text_chunks), desc="Ключевые слова"):
            result = future.result()
            keyword_counts.append(dict(result))

    combined_counter = Counter()
    for kw_dict in keyword_counts:
        combined_counter.update(kw_dict)

    return combined_counter.most_common(top_n)

def process_file(file_path: str, max_workers: int = None) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    models = get_models()
    _ = models.model
    _ = models.tokenizer
    _ = models.spacy_nlp

    chunks = split_text_by_sentences(text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_summary = executor.submit(parallel_summarize_chunks, chunks, max_workers)
        future_keywords = executor.submit(parallel_extract_keywords, [text], max_workers)

        chunk_summaries = future_summary.result()
        keywords = future_keywords.result()

    return {
        "keywords": keywords,
        "summary": " ".join(chunk_summaries),
        "chunk_summaries": chunk_summaries
    }

def save_structured_text(result: dict, output_base_path: str):
    with open(output_base_path + '.html', 'w', encoding='utf-8') as file:
        file.write('<div class="keywords-block">\n<div class="section-title">КЛЮЧЕВЫЕ СЛОВА:</div>\n')
        file.writelines(f'<span class="keyword-item">{phrase} <span class="keyword-count">({count})</span></span>\n'
                       for phrase, count in result["keywords"])
        file.write('</div>\n\n<div class="summary-block">\n<div class="section-title">СУММАРИЗАЦИЯ:</div>\n')
        file.writelines(f'<p>{summary}</p>\n' for summary in result["chunk_summaries"])
        file.write('</div>\n')

    with open(output_base_path + '.txt', 'w', encoding='utf-8') as file:
        file.write("=== КЛЮЧЕВЫЕ СЛОВА ===\n")
        file.writelines(f"- {phrase} (встречается {count} раз)\n" for phrase, count in result["keywords"])
        file.write("\n=== СУММАРИЗАЦИЯ ===\n")
        file.writelines(f"{summary}\n\n" for summary in result["chunk_summaries"])
