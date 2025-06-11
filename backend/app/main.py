from fastapi import FastAPI
from app.routes import auth, upload, synthgen, download

app = FastAPI(title="Medsynx API")

app.include_router(auth.router, prefix="/auth")
app.include_router(upload.router, prefix="/upload")
app.include_router(synthgen.router, prefix="/synth")
app.include_router(download.router, prefix="/download")
