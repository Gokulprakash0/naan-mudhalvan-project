import argparse
import fitz  # PyMuPDF
from transformers import pipeline
import pyttsx3
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def summarize_text(text, model_name="facebook/bart-large-cnn", chunk_size=1000, min_length=40, max_length=150):
    summarizer = pipeline("summarization", model=model_name)
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if not chunk:
            continue
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries) if summaries else "[No content to summarize]"

def speak_text(text, rate=150, save_audio_path=None):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    if save_audio_path:
        engine.save_to_file(text, save_audio_path)
        engine.runAndWait()
        print(f"✅ Audio saved to {save_audio_path}")
    else:
        engine.say(text)
        engine.runAndWait()

def save_summary(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✅ Summary saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="🔊 Convert a PDF to summarized speech using NLP.")
    
    # Required
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    
    # Optional
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for summarization (default: 1000).")
    parser.add_argument("--min_length", type=int, default=40, help="Minimum summary length per chunk (default: 40).")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum summary length per chunk (default: 150).")
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn", help="Hugging Face summarization model.")
    parser.add_argument("--rate", type=int, default=150, help="Speech rate in words per minute (default: 150).")
    parser.add_argument("--save_summary", type=str, help="Path to save the text summary.")
    parser.add_argument("--save_audio", type=str, help="Path to save the audio file.")

    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"❌ Error: File not found - {args.pdf_path}")
        return

    print("📄 Extracting text from PDF...")
    full_text = extract_text_from_pdf(args.pdf_path)

    if not full_text.strip():
        print("❌ Error: No text could be extracted from the PDF.")
        return

    print("🧠 Summarizing text...")
    summary = summarize_text(
        full_text,
        model_name=args.model,
        chunk_size=args.chunk_size,
        min_length=args.min_length,
        max_length=args.max_length
    )

    if args.save_summary:
        save_summary(summary, args.save_summary)

    print("🔈 Converting summary to speech...")
    speak_text(summary, rate=args.rate, save_audio_path=args.save_audio)

    print("✅ Done!")

if __name__ == "__main__":
    main()
