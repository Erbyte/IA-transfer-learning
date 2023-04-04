<h1 align="center">FLASK API - IA Transfer Learning Python</h1>
<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,tensorflow,flask" />
  </a>
</p>
<hr>

## Introduccion
Proyecto de transferencia de aprendizado usando modelo MobileNetV2, reemplazando la ultima capa por una
modificada para predecir entre tres elementos (Mouse, Teclado y Audifonos), por medio de una URL de una imagen,
la cual es enviada en formato JSON a una API basica en FLASK por medio del metodo POST.

## JSON
```
{
    "url": "url_de_la_imagen_a_predecir"
}
```
## Ejecucion
Para iniciar el servicio, es necesario ejecutar el programa atravez del archivo test.py. Este contiene la carga del
modelo preentrenado con la capa personalizada capaz de predecir entre los elementos anteriormente descritos.

Flask levanta un ambiente local con la ruta /predict en la cual se envia el JSON a traves del metodo POST.

```
POST: localhost:5000/predict
```

## Respuesta
La respuesta esperada para esta solicitud es en formato JSON y nos indica a que categoria pertenece el elemento
de la imagen enviada.

- 0: Audifonos 
- 1: Teclados 
- 2: Mouses

La estructura de la respuesta es la siguiente:
```
{
    "categoria": 0
}
```

## Entrenamiento
Para el entrenamiento del modelo se utilizaron 404 imagenes para cada categoria, las cuales fueron alteradas
con ImageDataGenerator estirandolas, rotando, agrandando, achicando y moviendo. De estas imagenes se
utilizaron un 20% para las pruebas.

Para crear el modelo, se congelaron las capas de MobileNet V2 para no perder el entrenamientos de estas y se
agrego una capa densa con activacion softmax, la cual es la que se entrenara para la prediccion.

Finalmente el entrenamiento se realizo en 50 epocas y el modelo exportado al final del codigo es el utilizado en
nuestra API basica realizada en FLASK.

<hr>

## Links
- <a href="https://www.youtube.com/watch?v=9Dur_oUMGG8&ab_channel=RingaTech">¿Pocos datos de entrenamiento? Prueba esta técnica</a>
- <a href="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4">Mobilenet V2 Feature Vector</a>

