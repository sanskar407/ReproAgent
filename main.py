"""
ReproAgent - FastAPI server
Serves the React frontend and provides API endpoints for the backend logic.
This replaces the Gradio UI for the Hugging Face Spaces Docker deployment.
"""

import os
import sys
import json
import uuid
import shutil
import traceback
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# --- App Setup ---
app = FastAPI(title="ReproAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes (must be defined BEFORE static file mounting) ---

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/easy-mode")
async def easy_mode(file: UploadFile = File(...)):
    """
    Easy Mode: Upload a PDF → Get AI summary + PPT download URL.
    Uses Gemini API for analysis.
    """
    tmp_dir = Path("data/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / f"{uuid.uuid4()}.pdf"

    try:
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Import the modular handlers
        from server.pdf_processor import extract_text_from_pdf
        from server.llm_handler import generate_summary_and_ppt_content
        from server.ppt_generator import generate_ppt

        # 1. Extract text
        text = extract_text_from_pdf(str(pdf_path))
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # 2. Call Gemini
        result = generate_summary_and_ppt_content(text)
        
        # If the LLM handler returned an error description, let's notify the user
        description = result.get("description", "")
        if "Please check your GEMINI_API_KEY" in description:
            # Check if slides contain more info
            error_msg = description
            if result.get("slides") and len(result["slides"]) > 0:
                error_detail = result["slides"][0].get("content", [""])[0]
                error_msg += f" (Detail: {error_detail})"
            raise HTTPException(status_code=500, detail=error_msg)

        slides = result.get("slides", [])

        # 3. Generate PPT
        ppt_filename = f"summary_{uuid.uuid4().hex[:8]}.pptx"
        ppt_path = tmp_dir / ppt_filename
        generate_ppt(slides, str(ppt_path))

        return JSONResponse({
            "description": description,
            "ppt_url": f"/api/download/{ppt_filename}"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EASY MODE ERROR] {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


@app.get("/api/download/{filename}")
def download_file(filename: str):
    """Download a generated PPT file."""
    file_path = Path("data/tmp") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


# --- Serve React Frontend (must be LAST) ---
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    # Mount static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        """Catch-all route: serve index.html for all frontend routes."""
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not built. Run 'npm run build' in the frontend/ directory."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
