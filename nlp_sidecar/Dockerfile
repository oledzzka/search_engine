FROM python:3.11-slim

WORKDIR /app

# torch CPU-only to keep image lean; swap for GPU base image if needed
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download models at build time so container starts fast
RUN python - <<'EOF'
from natasha import NewsEmbedding, NewsMorphTagger, NewsDependencyParser, NewsNERTagger
_ = NewsEmbedding()
from sentence_transformers import SentenceTransformer
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
from gliner import GLiNER
GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
print("Models downloaded.")
EOF

COPY main.py .

VOLUME ["/config/filters"]

EXPOSE 8001
CMD ["python", "main.py"]
