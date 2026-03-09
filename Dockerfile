# Usa la imagen oficial de Python ligera
FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y fuerza modo sin buffer
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crea el directorio de trabajo
WORKDIR /app

# Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente y modelos
COPY . .

# Usamos Uvicorn corriendo en el puerto de la variable $PORT que inyecta Google Cloud Run
CMD uvicorn app_extended:app --host 0.0.0.0 --port ${PORT:-8080}
