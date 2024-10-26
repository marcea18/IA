#import os                      --- caso tal q tengas problema con el SO Windows
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.datasets import mnist
import numpy as np 
#import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

# Paso 2: Cargar el conjunto de datos MNIST (tiene 60 mil imágenes)
#Xtesr y y Test tuenen 10 mil imagenes de prueba.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#.........................................................   PRueba: Ver imágenes......???
# Paso 2: Visualizar las primeras X imágenes del conjunto de entrenamiento
print("PRobando visualización de imágenes")
for i in range(5):
    plt.figure(figsize=(2, 2))  # Tamaño de la figura
    plt.imshow(x_train[i], cmap='gray')  # Mostrar la imagen en escala de grises
    plt.title(f"Etiqueta: {y_train[i]}")  # Título con la etiqueta de la imagen
    plt.axis('off')  # Ocultar ejes
    plt.show()
print("Finalinzando prueba. línea 19 visualización de imágenes")
#..........................................................................................

# Paso 3: Preprocesar los datos
x_train = x_train.astype('float32') / 255  # Normalización
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28 * 28)     # Aplanar las imágenes
x_test = x_test.reshape(-1, 28 * 28)
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Paso 4: Definir el modelo MLP.. construcción red neuronal
model = models.Sequential() # Aquí se creó el modelo
model.add(layers.Dense(512, activation='sigmoid', input_shape=(28 * 28,)))
model.add(layers.Dense(256, activation='sigmoid'))
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='sigmoid'))# Expirementar ...................prueba.........
model.add(layers.Dense(32, activation='sigmoid')) 
model.add(layers.Dense(10, activation='softmax'))  # 10 clases de salida

# Paso 5: Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Paso 6: Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
#history graba lo que sucedió en las épocas
#....................................................... PRueba................ luego del entrenamiento
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(history.history["loss"])
# ...........................................................................................

# Paso 7: Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_acc}')
# Paso 8: Probar el modelo con una imagen del conjunto de prueba
imagen = x_test[0].reshape(1, 28 * 28)  # Seleccionar la primera imagen
prediccion = model.predict(imagen)
digit_predicho = np.argmax(prediccion)
# Mostrar la imagen y el dígito predicho
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f'Predicción: {digit_predicho}')
plt.show()

# A partir de aqui para carga una imágen de manera local
# Paso 9: Probar el modelo con una imagen personalizada
#ruta_imagen="D:/talento tech/ia/imagen1.png"
def predecir_imagen_personalizada(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28 píxeles
    img_array = np.array(img).reshape(1, 28 * 28).astype('float32') / 255  # Normalizar
    prediccion = model.predict(img_array)  # Hacer la predicción
    digit_predicho = np.argmax(prediccion)
    print(f'Predicción para la imagen personalizada (Imágen test2): {digit_predicho}')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicción: {digit_predicho}') #
    plt.show()
# Prueba con una imagen personalizada (comenta o descomenta según lo necesites)
predecir_imagen_personalizada('imagen.png')

#repositorio git hub










