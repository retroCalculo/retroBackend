# Usa la imagen oficial de Python 3.9
FROM python:3.9.10

# Establece el directorio de trabajo
WORKDIR /deployRetro/Api

# Copia los archivos de tu aplicación al directorio de trabajo
COPY . /deployRetro

# Copia los archivos de tu aplicación al directorio de trabajo
COPY ./Api /deployRetro/Api

# Instala las dependencias
RUN pip install --no-cache-dir -r /deployRetro/requirements.txt

# Expone el puerto en el que la aplicación se ejecutará (ajústalo según sea necesario)
EXPOSE 80

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
