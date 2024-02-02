from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import retro_calculo as data
import os

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "https://retro-calculo.vercel.app"}})


@app.route('/api/get_data', methods=['GET'])

def get_data():
    geofonos = data.geofonos()
    mensaje = {'geofonos': geofonos}
    return mensaje


@app.route('/api/get_tabla', methods=['GET'])

def get_tabla():
    tabla = data.tabla()
    return tabla

@app.route('/api/get_image1', methods=['GET'])
def get_image1():
    # Ruta a la imagen en tu servidor
    image_path = 'images/image1.jpg'

    # Verificar si la imagen existe
    if os.path.exists(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Content-Type'] = 'image/jpeg; charset=utf-8'

        return response
    else:
        return 'Imagen no encontrada', 404
    
@app.route('/api/get_image2', methods=['GET'])
def get_image2():
    # Ruta a la imagen en tu servidor
    image_path = 'images/image2.jpg'

    # Verificar si la imagen existe
    if os.path.exists(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Content-Type'] = 'image/jpeg; charset=utf-8'

        return response
    else:
        return 'Imagen no encontrada', 404
    
@app.route('/api/get_image3', methods=['GET'])
def get_image3():
    # Ruta a la imagen en tu servidor
    image_path = 'images/image3.jpg'

    # Verificar si la imagen existe
    if os.path.exists(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Content-Type'] = 'image/jpeg; charset=utf-8'

        return response
    else:
        return 'Imagen no encontrada', 404
    
@app.route('/api/get_image4', methods=['GET'])
def get_image4():
    # Ruta a la imagen en tu servidor
    image_path = 'images/image4.jpg'

    # Verificar si la imagen existe
    if os.path.exists(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Content-Type'] = 'image/jpeg; charset=utf-8'

        return response
    else:
        return 'Imagen no encontrada', 404
    
@app.route('/api/get_image5', methods=['GET'])
def get_image5():
    # Ruta a la imagen en tu servidor
    image_path = 'images/image5.jpg'

    # Verificar si la imagen existe
    if os.path.exists(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Content-Type'] = 'image/jpeg; charset=utf-8'

        return response
    else:
        return 'Imagen no encontrada', 404
    
@app.route('/api/get_image6', methods=['GET'])
def get_image6():
    # Ruta a la imagen en tu servidor
    image_path = 'images/image6.jpg'

    # Verificar si la imagen existe
    if os.path.exists(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Content-Type'] = 'image/jpeg; charset=utf-8'

        return response
    else:
        return 'Imagen no encontrada', 404
    
@app.route('/api/cargar_csv', methods=['POST'])

def cargar_csv():
    archivo_csv = request.files['archivo_csv']
    temperatura = request.form.get('temperatura')

    with open('temperatura.txt', 'w') as file:
        file.write(temperatura)
    
    # Procesa el archivo CSV utilizando pandas
    df = pd.read_csv(archivo_csv)

    df.to_pickle('dataframe.pkl')

    return jsonify({'mensaje': 'Datos del CSV procesados correctamente'})


@app.route('/api/distancia_geofono', methods=['POST'])

def distancia_geofono():
    distancia_D = request.form.get('distancia_D')
    with open('distancia_D.txt', 'w') as file:
        file.write(distancia_D)

    return jsonify({'mensaje': 'Distancia_D correctamente'})

if __name__ == '__main__':
        app.run(host="0.0.0.0", port=80, debug=True)
    
