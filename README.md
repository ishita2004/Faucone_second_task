# RAG PDF Intake UI

This project provides a simple user interface for scanning all PDF files from a local `pdfs` folder and copying them into a GitHub destination folder one by one.

## Structure

- `pdfs/` - source folder containing the PDF documents
- `github/` - destination folder where processed PDFs are copied
- `app.py` - Streamlit-based interface

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This version focuses on the UI and file-handling workflow only. It is ready to be connected to your actual RAG indexing or GitHub upload logic.
