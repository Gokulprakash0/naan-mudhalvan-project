# backend.py
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from mmain import extract_text_from_pdf, summarize_text, speak_text

app = FastAPI()

# Serve static files (e.g., back.html)
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.post("/summarize-pdf/")
async def summarize_pdf(
    pdf: UploadFile = File(...),
    save_audio: bool = Form(False)
):
    # Save the uploaded PDF
    pdf_path = f"temp_{pdf.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Extract text from the PDF
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return JSONResponse({"error": "No text could be extracted from the PDF."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to extract text: {str(e)}"}, status_code=500)

    # Summarize the text
    try:
        summary = summarize_text(text)
    except Exception as e:
        return JSONResponse({"error": f"Failed to summarize text: {str(e)}"}, status_code=500)

    # Optionally generate audio
    audio_file = None
    if save_audio:
        audio_file = "summary.mp3"
        try:
            speak_text(summary, save_audio_path=audio_file)
        except Exception as e:
            return JSONResponse({"error": f"Failed to generate audio: {str(e)}"}, status_code=500)

    # Clean up the temporary PDF file
    os.remove(pdf_path)

    # Return the summary and audio file path
    return {"summary": summary, "audio_file": audio_file}
