# Dockerfile (porta 8080 fixada)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar código
COPY main.py /app/main.py

# Porta (Render pode ignorar EXPOSE, mas local ajuda)
EXPOSE 8080

# Forçar porta 8080
CMD uvicorn main:app --host 0.0.0.0 --port 8080
