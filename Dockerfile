FROM python:3.11-slim

# Sin Tesseract, sin libGL, sin PyTorch. Al mover los motores a sus propios
# servicios este contenedor pasó de ~2,5 GB a ~200 MB y arranca en segundos.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

# `HOST` y `PORT` salen del entorno para no tener que rearmar la imagen al
# desplegar. En Railway `PORT` lo inyecta la plataforma; `HOST` sólo hay que
# tocarlo si el servicio tiene que escuchar en IPv6 (`::`) para la red privada.
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8003}"]
