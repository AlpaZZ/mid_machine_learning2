# 1. Base Image menggunakan Python slim yang ringan
FROM python:3.11-slim

# 2. Atur environment variables untuk kestabilan Python di container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# 3. Buat direktori kerja di dalam container
WORKDIR /app

# 4. Install tool pendukung sistem jika diperlukan (seperti curl/git)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 5. Salin berkas dependencies dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Salin seluruh isi proyek ke dalam container
COPY . .

# 7. Expose port Streamlit
EXPOSE 8501

# 8. Konfigurasi healthcheck untuk monitoring container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Perintah untuk menjalankan Streamlit di production
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
