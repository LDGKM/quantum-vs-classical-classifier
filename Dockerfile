# Image de l'application web, utilisée par Hugging Face Spaces (SDK Docker).
FROM python:3.12-slim

# Hugging Face exécute le conteneur avec l'utilisateur 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLBACKEND=Agg
WORKDIR /home/user/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user app.py .
COPY --chown=user src/ src/
COPY --chown=user .streamlit/ .streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
