# Build from the project root:
#   docker build -t soccer-xg-app -f app/Dockerfile .
# Run:
#   docker run -p 8501:8501 soccer-xg-app

FROM python:3.12-slim

WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# The app needs only the code, the trained model, the compact shots file
# (app/data/) and the analysis results - no raw data or download.
COPY src/ src/
COPY app/ app/
COPY models/xg_model.joblib models/xg_model.joblib
COPY results/ results/

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
