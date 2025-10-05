FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY poetry.lock ./
RUN pip install --upgrade pip && pip install poetry
RUN poetry install --no-interaction --no-ansi

COPY src/ src/
COPY tests/utils/ tests/utils/

EXPOSE 8501

# Ensure Streamlit binds to all interfaces and uses correct port
CMD ["streamlit", "run", "src/demo_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]