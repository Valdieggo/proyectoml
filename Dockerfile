# Usa una imagen base de Python
FROM python:3.10.12

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY models models
COPY ejecmodels.py .
COPY atp_matches_2023.csv .
COPY requirements.txt .


# Instala las dependencias (puedes ajustar según tus necesidades)
RUN pip install --no-cache-dir -r requirements.txt

# Ejecuta tu script al iniciar el contenedor
CMD ["python", "ejecmodels.py"]
