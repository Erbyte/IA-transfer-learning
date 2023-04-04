import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub


# Aumento de datos con ImageDataGenerator
# Crear el generador
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=15,
    zoom_range=[0.5, 1.5],
    validation_split=0.2  # 20% para pruebas
)

# Generadores para sets de entrenamiento y pruebas
data_gen_entrenamiento = datagen.flow_from_directory('datasets', target_size=(224, 224), batch_size=32, shuffle=True,
                                                     subset='training')
data_gen_pruebas = datagen.flow_from_directory('datasets', target_size=(224, 224), batch_size=32, shuffle=True,
                                               subset='validation')

# Imprimir 10 imagenes de entrenamiento
for imagen, etiqueta in data_gen_entrenamiento:
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i])
    break

plt.show()

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# Crear modelo
modelo = tf.keras.Sequential([
    hub.KerasLayer(url, input_shape=(224, 224, 3), trainable=False),
    tf.keras.layers.Dense(3, activation='softmax')
])

modelo.summary()

# Compilar modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
epochs = 50
historial = modelo.fit(data_gen_entrenamiento, epochs=epochs, batch_size=32, validation_data=data_gen_pruebas)

# Exportar modelo en formato h5
modelo.save('model/modelo_transfer_learning.h5')

# Exportar con tensorflow JS para web (terminal)
# tensorflowjs_converter --input_format keras modelo.h5 carpeta_salida