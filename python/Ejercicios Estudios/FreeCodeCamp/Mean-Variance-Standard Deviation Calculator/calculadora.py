# Cargamos las librerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate(list):
    try:
        matriz=np.array(list)
        matriz=np.reshape(matriz,(3,3))
        mean_flat=np.mean(matriz)
        mean_axis_0=np.mean(matriz,axis=0)
        mean_axis_1=np.mean(matriz,axis=1)
        variance_flat=np.var(matriz)
        variance_axis_0=np.var(matriz,axis=0)
        variance_axis_1=np.var(matriz,axis=1)
        std_flat=np.std(matriz)
        std_axis_0=np.std(matriz,axis=0)
        std_axis_1=np.std(matriz,axis=1)
        max_flat=np.max(matriz)
        max_axis_0=np.max(matriz,axis=0)
        max_axis_1=np.max(matriz,axis=1)
        min_flat=np.min(matriz)
        min_axis_0=np.min(matriz,axis=0)
        min_axis_1=np.min(matriz,axis=1)
        sum_flat=np.sum(matriz)
        sum_axis_0=np.sum(matriz,axis=0)
        sum_axis_1=np.sum(matriz,axis=1)
        calculations={
            "Media":[mean_axis_0.tolist(),mean_axis_1.tolist(),mean_flat],
            "Varianza":[variance_axis_0.tolist(),variance_axis_1.tolist(),variance_flat],
            "Desviacion estandar":[std_axis_0.tolist(),std_axis_1.tolist(),std_flat],
            "Maximo":[max_axis_0.tolist(),max_axis_1.tolist(),max_flat],
            "Minimo":[min_axis_0.tolist(),min_axis_1.tolist(),min_flat],
            "Suma":[sum_axis_0.tolist(),sum_axis_1.tolist(),sum_flat]
            }
        return calculations
    except ValueError:
        print('List must contain nine numbers.')

print(calculate([0,1,2,3,4,5,6,7,8]))
#calculations.shape