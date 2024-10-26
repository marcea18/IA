import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#CARGAR MODELO YA ENTRENADO
model = tf.keras.models.load_model('/Users/marcea/Documents/CursoIA-Integrador/Nivel2/TALLER1')

#Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28 píxeles
    image = np.array(image) #convertir a array numpy
    image = image / 255.0 #Normalizar valores entre 0 y 1
    image = np.reshape(image,(1, 28*28)) #Redimensionar para la entrada del modelo (1= representa el 3 de imagen)
                                                                                            #(28, 28= pixeles de sncho y alto)
                                                                                            #(1 significa escala de grises, escala rgb se coloca un 3)    
    return image
#Titulo de la aplicación

st.title('Clasificación de dígitos manuscritos')

#Cargar la imagen
uploaded_file = st.file_uploader('Cargar una imagen de un dígito (0,9)', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    #Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)
    
    #Preprocesar la imagen
    processed_image = preprocess_image(image)
    
    #Hacer la predicción
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    #Mostrar la predicción
    st.write(f'Predicción:  **{predicted_digit}**')
    
    #Mostrar probabilidades
    for i in range(10):
        st.write(f'Dígito {i}: {prediction[0][i]:.4f}')
        
    #Mostrar imagen procesada para cer el preprocesamiento
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')  # Convierte a 28x28 para visualizar
    plt.axis('off')
    st.pyplot(plt)
    
    

















