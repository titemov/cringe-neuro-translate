import pandas as pd
import torch
import re
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


def ms_to_srt_time(ms):
    ms = int(ms)

    hours = ms // 3600000
    ms %= 3600000

    minutes = ms // 60000
    ms %= 60000

    seconds = ms // 1000
    milliseconds = ms % 1000

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_srt_with_adjustment(df, output_file):
    n = len(df)

    # Подгонка окончаний
    for i in range(n - 1):
        if df.loc[i, "end"] < df.loc[i + 1, "start"]:
            df.loc[i, "end"] = df.loc[i + 1, "start"]

    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            start = ms_to_srt_time(row["start"])
            end = ms_to_srt_time(row["end"])
            text = row["translated"]

            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def is_text_too_short(text, min_len):
    cleaned = re.sub(r"[^\w]", "", text)
    return len(cleaned) < min_len


def compress_text(text, tokenizer, model, device):
    """
    Сжатие текста через ruT5
    """
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=10,
            min_length=1,
            length_penalty=2.0,
            num_beams=4
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def translate_tsv_to_srt(
        input_file,
        output_file,
        model_name="Helsinki-NLP/opus-mt-en-ru",
        batch_size=8,
        min_len=4,
        fallback_text="негры негры"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ===== Переводчик =====
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    print("Translation model loaded")

    # ===== Модель сжатия =====
    print("Loading compression model...")

    sum_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-base-absum")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("cointegrated/rut5-base-absum").to(device)

    print("Compression model loaded")

    # ===== Чтение TSV =====
    df = pd.read_csv(input_file, sep="\t")

    if "text" not in df.columns:
        raise ValueError("TSV файл должен содержать колонку 'text'")

    texts = df["text"].fillna("").astype(str).tolist()

    translated_texts = []

    print("Translating...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            translated = model.generate(**inputs)

        batch_translations = [
            tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]

        translated_texts.extend(batch_translations)

    print("Post-processing (compression)...")

    fixed_translations = []

    for text in translated_texts:
        text = text.strip()

        if is_text_too_short(text, min_len):
            text = fallback_text
        else:
            try:
                text = compress_text(text, sum_tokenizer, sum_model, device)
            except Exception as e:
                print("Compression error:", e)
                # fallback — оставить перевод
                pass

        fixed_translations.append(text)

    df["translated"] = fixed_translations

    print("Writing SRT...")
    write_srt_with_adjustment(df, output_file)

    print("Done:", output_file)


if __name__ == "__main__":
    print("enter filename")

    userfilename = str(input())

    translate_tsv_to_srt(
        input_file=userfilename,
        output_file=f"{userfilename}_ru.srt",
        batch_size=8,
        min_len=4,
        fallback_text="подпишись на лалаласкул"
    )