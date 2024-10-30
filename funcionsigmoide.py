#Función Sigmoide predición de aprobación de crédito, caso puntaje de riesgo crediticio

import numpy as np
import matplotlib.pyplot as plt

#función sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Datos de ejemplo para puntajes de crédito de 10 clientes
credit_score = np.array([-8,-4,0,2,5,7,-6,3,10,2])

#Aplicar la función sigmoide
approved_probabilities = sigmoid(credit_score)

# Visualizamos las probabilidades de aprobación de crédito
plt.plot(credit_score, approved_probabilities, marker='o',linestyle='-',label='Probabilidad de Aprobación')
plt.title('Probabilidad de Aprobación del Crédito usando sigmoide')
plt.xlabel('Puntaje de Crédito')
plt.ylabel('Probabilidad de Aprobación')
plt.legend()
plt.grid(True)
plt.show()

print("probabilidad de aprobación",approved_probabilities)