# ---- imagem base ----
FROM python:3.11-slim

# Evita bytecode e força logs imediatos
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependências do sistema (opcional mas ajuda em debug)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tini && \
    rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copia dependências primeiro (melhor cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia a aplicação
COPY main.py /app/main.py

# Porta (Render usa $PORT; local usa 10000)
EXPOSE 10000

# Saúde rápida (opcional): você pode usar /health
# HEALTHCHECK CMD curl -fsS http://localhost:10000/health || exit 1

# Entrypoint com tini (mata zumbis) e comando padrão uvicorn
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-10000}"]
