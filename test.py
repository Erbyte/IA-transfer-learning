import tensorflow as tf
from io import BytesIO
import requests
from PIL import Image
import cv2
import numpy as np
import tensorflow_hub as hub
from flask import Flask, jsonify, request

# Cargar modelo
modelo = tf.keras.models.load_model('model/modelo_transfer_learning.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Crear instancia de Flask
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data['url']
    categoria = categorizar(url)
    output = {'categoria': int(categoria)}
    return jsonify(output)


def categorizar(url):
    respuesta = requests.get(url)
    img = Image.open(BytesIO(respuesta.content))
    img = np.array(img).astype(float) / 255

    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)


if __name__ == '__main__':
    app.run()

# Debe ser 1
# predict_teclado = categorizar('https://androidpc.es/wp-content/uploads/2021/03/teclado-ingles-internacional-n00.jpg')
# print(predict_teclado)

# Debe ser 2
# predict_mouse = categorizar('https://d1pc5hp1w29h96.cloudfront.net/magefan_blog/mouse_ptico_vs_l_ser.jpg')
# print(predict_mouse)

# Debe ser 0
# predict_headphones = categorizar(
# 'https://helios-i.mashable.com/imagery/reviews/01HUTTQhSs8SWLx3ouc0f7q/hero-image.fill.size_1200x1200.v1653070267.jpg')
# print(predict_headphones)
