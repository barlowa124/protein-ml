FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" \
    && pip cache purge

COPY data/processed/ddpm.pt /model/ddpm.pt
COPY data/processed/gb1.parquet /model/gb1.parquet
ENV MODEL_PATH=/model/ddpm.pt \
    LANDSCAPE_PATH=/model/gb1.parquet \
    PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "protein_diffusion.serve:app", "--host", "0.0.0.0", "--port", "8000"]
