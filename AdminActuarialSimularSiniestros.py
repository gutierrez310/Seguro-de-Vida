# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:17:21 2020

@author: carlo
"""


import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# Esta direccion tiene que apuntar directamente a la localizacion de las tablas de mortalidad.
# Estas tablas de mortalidad tienen que tener el rango apropiado en cuanto a edades
path = 'D:\Material academico\9no semestre\Administracion Actuarial\ParaSimularSiniestralidad.csv'
    # https://www.ssa.gov/oact/STATS/table4c6.html
file = open(path,newline='')
reader = csv.reader(file)
header = next(reader) #primera linea con columnas

data = []
for row in reader:
    data.append([int(row[0]),float(row[1])])
I = 0
np.random.seed(I)
EDAD_PROMEDIO = 40
DESV_ESTANDAR1 = 20
TAMAÑO_MUESTRA = 3000
muestra = np.random.uniform(15, 80, TAMAÑO_MUESTRA)

def allocate(minO,maxO,minP,maxP,x):
    return minO+(x-minP)/(maxP-minP)*(maxO-minO)

minP = min(muestra)
maxP = max(muestra)
minO = 15
maxO = 80

muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))


plt.hist(muestra1, bins = 65)
plt.show()


siniestralidad = []

def sim_qx(prob):
    """
    prob : Una probabilidad , 0 < prob <= 1
    
    Esta funcion simula la muerte con probabilidad "prob"
    Regresa True si simula muerte
    Y False si simula supervivencia
    """
    resultado = int(np.random.choice([0,1],1,p=[1-prob,prob])[0])
    if resultado == 1:
        return True
    else:
        return False


data[-1][1]=1

def d_x(x):
    return data[x][1]*l_x_list[x]

l_x_list=[1000000]
for i in range(1,len(data)+1,1):
    l_x_list.append(l_x_list[i-1] - d_x(i-1))


def n1Qx(n,x):
    """
    n| q_x = d_(x+n) / l_x
    """
    return d_x(x+n)/l_x_list[x]

simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))

summ = 0
for i in simulacion:
    summ+=1

print("Exposicion Total: ", summ)


dict_simulacion={}
for i in simulacion:
    """
    'Edad' : ['Exitos','Fracasos']
    """
    if i[0] not in dict_simulacion:
        dict_simulacion[i[0]]=[0,0]
    if i[1]:
        dict_simulacion[i[0]][0]+=1
    else:
        dict_simulacion[i[0]][1]+=1


rounded_up_muestra1= list(map(lambda x: math.ceil(x),muestra1))
rounded_up_muestra1.sort()
k = []
n_x = [] 
q_x = []
Ex = []
varX = []
ES = []
varS = []
# b_x = 
temp=0
for i in sorted(dict_simulacion.keys()):
    k.append(i)
    n_x.append(rounded_up_muestra1.count(i))
    q_x.append(dict_simulacion[i][0]/ (dict_simulacion[i][1]+dict_simulacion[i][0]))
    Ex.append(q_x[temp])
    varX.append(q_x[temp]*(1-q_x[temp]))
    ES.append(Ex[temp]*n_x[temp])
    varS.append(q_x[temp]*(1-q_x[temp])*n_x[temp])
    temp += 1

#Teta la tasa para el calculo de las primas
#VarS es un arreglo de varianzas por edades, prob*(1-prob)*n
#ES es un arreglo de valor esperado por edades, prob*n
print("\nTeta = 1.645*var(s)^0.5/E[S] , al 95%")
teta = (1.645*sum(varS)**0.5)/sum(ES)
print("Teta = ", teta)


prima_total = (teta+1)*sum(ES)
print("Prima de la cartera: ", prima_total)
prima = prima_total/TAMAÑO_MUESTRA
print("Prima por individuo promedio: ", prima)



np.random.seed(I+1000)
DESV_ESTANDAR2 = 14
# TAMAÑO_MUESTRA = 1000000
#datos aleatorios, RAW
muestra = np.random.gamma(120,5, TAMAÑO_MUESTRA)

minP = min(muestra)
maxP = max(muestra)
minO = 15
maxO = 80

#muestra las edad muestrales, acomodadas al rango
muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))

plt.hist(muestra1, bins = 65)
plt.show()


# muestra la edad, y si murio o no
simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))

summ = 0
for i in simulacion:
    if i[1]:
        summ+=1

print("Prima obtenida: ", prima*TAMAÑO_MUESTRA)
print("Siniestralidad Total: ", summ)
print("Ganancias: ", prima*TAMAÑO_MUESTRA - summ)

# =============================================================================
# print("a) R= La prima calculada por individuo promedio es de: ", prima_total/TAMAÑO_MUESTRA)
# 
# 
# print("b) Calcular nuevo costo de prima con B variable...")
# =============================================================================




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Parte 1

#Como fue platica con el profesor durante clase, la distribucion de la suma asegurada
#en la poblacion general no es constante, mas la hemos estimado a un modelo chi^2 
#donde la poblacion central tiene una suma asegurada mayor
from scipy.stats import chi2
costos = list(chi2.pdf(range(len(k)),25))
for i in range(len(costos)):
    costos[i] *= 1000
    costos[i] += 5
print("CASO B variante, ")
print("\n\nCostos aleatorios: ")
print("Min: ", min(costos))
print("Promedio: ", sum(costos)/len(costos))
print("Max: ", max(costos))

k = []
n_x = [] 
q_x = []
Ex = []
varX = []
ES = []
varS = []
b_x = []
temp=0
for i in sorted(dict_simulacion.keys()):
    k.append(i)
    b_x.append(costos[temp])
    n_x.append(rounded_up_muestra1.count(i))
    q_x.append(dict_simulacion[i][0]/ (dict_simulacion[i][1]+dict_simulacion[i][0]))
    Ex.append(q_x[temp]*b_x[temp])
    varX.append(q_x[temp]*(1-q_x[temp])*b_x[temp]**2)
    ES.append(Ex[temp]*n_x[temp])
    varS.append(varX[temp]*n_x[temp])
    temp += 1
    
print("\nTeta = 2.3263*var(s)^0.5/E[S] , al 99%")
teta = (2.3263*sum(varS)**0.5)/sum(ES)
print("Teta = ", teta)

prima_total = (teta+1)*sum(ES)
print("Prima de la cartera: ", prima_total)
prima = prima_total/TAMAÑO_MUESTRA
print("Prima por individuo promedio: ", prima_total/TAMAÑO_MUESTRA)







np.random.seed(I+1000)
DESV_ESTANDAR2 = 14
#TAMAÑO_MUESTRA = 1000
muestra = np.random.gamma(120,5, TAMAÑO_MUESTRA)

minP = min(muestra)
maxP = max(muestra)
minO = 15
maxO = 80

muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))

#Siniestralidad Total
plt.hist(muestra1, bins = 65)
plt.show()



simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))

siniestralidad_total= 0
for i in range(len(simulacion)):
    if simulacion[i][1]:
        siniestralidad_total+=costos[simulacion[i][0]-15]

prima_obtenida = prima*TAMAÑO_MUESTRA
ganancias = prima_obtenida - siniestralidad_total
print("Prima obtenida: ", prima_obtenida)
print("Siniestralidad Total: ", siniestralidad_total)
print("Ganancias: ", prima_obtenida - siniestralidad_total)
print("\n\n\n")



primas = []
temp=0
for i in sorted(dict_simulacion.keys()):
    primas.append(ES[temp]*(1 + teta))
    temp += 1

summ=0
print("Edad, n_x, prima_x, prima_s")
for i in k:
    print(i,n_x[i-15],primas[i-15]/n_x[i-15],primas[i-15])
    summ+=primas[i-15]




print("Prima obtenida: ", summ)
print("Siniestralidad Total: ", siniestralidad_total)
print("Ganancias: ", summ - siniestralidad_total)

siniestralidad_arreglo = []
for I in range(1,1001):
    np.random.seed(I)
    #EDAD_PROMEDIO = 40
    #DESV_ESTANDAR = 20
    #TAMAÑO_MUESTRA = 1000
    muestra = np.random.uniform(15, 80, TAMAÑO_MUESTRA)
    minP = min(muestra)
    maxP = max(muestra)
    minO = 15
    maxO = 80
    muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))
    siniestralidad = []
    simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))
    dict_simulacion={}
    for i in simulacion:
        if i[0] not in dict_simulacion:
            dict_simulacion[i[0]]=[0,0]
        if i[1]:
            dict_simulacion[i[0]][0]+=1
        else:
            dict_simulacion[i[0]][1]+=1
    rounded_up_muestra1= list(map(lambda x: math.ceil(x),muestra1))
    rounded_up_muestra1.sort()
    k = []
    n_x = [] 
    q_x = []
    Ex = []
    varX = []
    ES = []
    varS = []
    # b_x = 
    temp=0
    for i in sorted(dict_simulacion.keys()):
        k.append(i)
        n_x.append(rounded_up_muestra1.count(i))
        q_x.append(dict_simulacion[i][0]/ (dict_simulacion[i][1]+dict_simulacion[i][0]))
        Ex.append(q_x[temp])
        varX.append(q_x[temp]*(1-q_x[temp]))
        ES.append(Ex[temp]*n_x[temp])
        varS.append(q_x[temp]*(1-q_x[temp])*n_x[temp])
        temp += 1
    # print("\nTeta = 1.645*var(s)^0.5/E[S] , al 95%")
    teta = (1.645*sum(varS)**0.5)/sum(ES)
    # print("Teta = ", teta)
    prima_total = (teta+1)*sum(ES)
    # print("Prima de la cartera: ", prima_total)
    prima = prima_total/TAMAÑO_MUESTRA
    # print("Prima por individuo promedio: ", prima)
    np.random.seed(I)
    # DESV_ESTANDAR2 = 14
    # TAMAÑO_MUESTRA = 1000000
    muestra = np.random.gamma(120,5, TAMAÑO_MUESTRA)
    minP = min(muestra)
    maxP = max(muestra)
    minO = 15
    maxO = 80
    muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))
    #plt.hist(muestra1, bins = 65)
    #plt.show()
    simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))
    summ = 0
    for i in simulacion:
        if i[1]:
            summ+=1
# =============================================================================
#     print("Prima obtenida: ", prima*TAMAÑO_MUESTRA)
#     print("Siniestralidad Total: ", summ)
# =============================================================================
    siniestralidad_arreglo.append(summ)
    print("iteracion:", I)
# =============================================================================
#     print("Ganancias: ", prima*TAMAÑO_MUESTRA - summ)
#     costos = list(chi2.pdf(range(len(k)),25))
#     for i in range(len(costos)):
#         costos[i] *= 1000
#         costos[i] += 5
#     print("\n\nCostos aleatorios: ")
#     print("Min: ", min(costos))
#     print("Promedio: ", sum(costos)/len(costos))
#     print("Max: ", max(costos))
#     k = []
#     n_x = [] 
#     q_x = []
#     Ex = []
#     varX = []
#     ES = []
#     varS = []
#     b_x = []
#     temp=0
#     for i in sorted(dict_simulacion.keys()):
#         k.append(i)
#         b_x.append(costos[temp])
#         n_x.append(rounded_up_muestra1.count(i))
#         q_x.append(dict_simulacion[i][0]/ (dict_simulacion[i][1]+dict_simulacion[i][0]))
#         Ex.append(q_x[temp]*b_x[temp])
#         varX.append(q_x[temp]*(1-q_x[temp])*b_x[temp]**2)
#         ES.append(Ex[temp]*n_x[temp])
#         varS.append(varX[temp]*n_x[temp])
#         temp += 1   
#     print("\nTeta = 2.3263*var(s)^0.5/E[S] , al 99%")
#     teta = (2.3263*sum(varS)**0.5)/sum(ES)
#     print("Teta = ", teta)
#     prima_total = (teta+1)*sum(ES)
#     print("Prima de la cartera: ", prima_total)
#     prima = prima_total/TAMAÑO_MUESTRA
#     print("Prima por individuo promedio: ", prima_total/TAMAÑO_MUESTRA)
#     np.random.seed(I+1000)
#     DESV_ESTANDAR = 14
#     #TAMAÑO_MUESTRA = 1000
#     muestra = np.random.gamma(120,5, TAMAÑO_MUESTRA)
#     minP = min(muestra)
#     maxP = max(muestra)
#     minO = 15
#     maxO = 80
#     muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))
#     simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))
#     siniestralidad_total= 0
#     for i in range(len(simulacion)):
#         if simulacion[i][1]:
#             siniestralidad_total+=costos[simulacion[i][0]-15]
#     prima_obtenida = prima*TAMAÑO_MUESTRA
#     ganancias = prima_obtenida - siniestralidad_total
#     print("CASO B variante, ")
#     print("Prima obtenida: ", prima_obtenida)
#     print("Siniestralidad Total: ", siniestralidad_total)
#     print("Ganancias: ", prima_obtenida - siniestralidad_total)
#     print("\n\n\n")
#     primas = []
#     temp=0
#     for i in sorted(dict_simulacion.keys()):
#         primas.append(ES[temp]*(1 + teta))
#         temp += 1
#     summ=0
#     print("Edad, n_x, prima_x, prima_s")
#     for i in k:
#         print(i,n_x[i-15],primas[i-15]/n_x[i-15],primas[i-15])
#         summ+=primas[i-15]
#     print("Prima obtenida: ", summ)
#     print("Siniestralidad Total: ", siniestralidad_total)
#     print("Ganancias: ", summ - siniestralidad_total)
# =============================================================================


promedio=np.mean(siniestralidad_arreglo)
mediana=np.median(siniestralidad_arreglo)
minimo=min(siniestralidad_arreglo)
maximo=max(siniestralidad_arreglo)
desv_est=np.std(siniestralidad_arreglo)
print("promedio", promedio)
print("mediana", mediana)
print("minimo", minimo)
print("maximo", maximo)
print("desv_est", desv_est)

print("muertes totales",sum(siniestralidad_arreglo))


teta = (1.645*desv_est)/sum(siniestralidad_arreglo)
print("teta", teta)
plt.hist(siniestralidad_arreglo,bins=27)
plt.show()

import scipy.stats as ss
print("Normalidad por la funcion 'scipy.stat.normaltest':\n", ss.normaltest(siniestralidad_arreglo))

prima_general=(1+teta)*(0.13246049999999995)/(1-.04)


#Demostrar el 95% de confianza


ganancias=[]
for I in range(10000,10000+100):
    np.random.seed(I)
    #EDAD_PROMEDIO = 40
    #DESV_ESTANDAR = 20
    #TAMAÑO_MUESTRA = 1000
    muestra = np.random.uniform(15, 80, TAMAÑO_MUESTRA)
    minP = min(muestra)
    maxP = max(muestra)
    minO = 15
    maxO = 80
    muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))
    siniestralidad = []
    simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))
    dict_simulacion={}
    for i in simulacion:
        if i[0] not in dict_simulacion:
            dict_simulacion[i[0]]=[0,0]
        if i[1]:
            dict_simulacion[i[0]][0]+=1
        else:
            dict_simulacion[i[0]][1]+=1
    rounded_up_muestra1= list(map(lambda x: math.ceil(x),muestra1))
    rounded_up_muestra1.sort()
    k = []
    n_x = [] 
    q_x = []
    Ex = []
    varX = []
    ES = [1]
    varS = []
    # b_x = 
    temp=0
    for i in sorted(dict_simulacion.keys()):
        k.append(i)
        n_x.append(rounded_up_muestra1.count(i))
        q_x.append(dict_simulacion[i][0]/ (dict_simulacion[i][1]+dict_simulacion[i][0]))
        Ex.append(q_x[temp])
        varX.append(q_x[temp]*(1-q_x[temp]))
        ES.append(Ex[temp]*n_x[temp])
        varS.append(q_x[temp]*(1-q_x[temp])*n_x[temp])
        temp += 1
    # print("\nTeta = 1.645*var(s)^0.5/E[S] , al 95%")
    teta = (1.645*sum(varS)**0.5)/sum(ES)
    # print("Teta = ", teta)
    prima_total = (teta+1)*sum(ES)
    # print("Prima de la cartera: ", prima_total)
    prima = prima_general#prima_total/TAMAÑO_MUESTRA
    # print("Prima por individuo promedio: ", prima)
    np.random.seed(I)
    # DESV_ESTANDAR2 = 14
    # TAMAÑO_MUESTRA = 1000000
    muestra = np.random.gamma(120,5, TAMAÑO_MUESTRA)
    minP = min(muestra)
    maxP = max(muestra)
    minO = 15
    maxO = 80
    muestra1=list(map(lambda x: allocate(minO,maxO,minP,maxP,x),muestra))

    simulacion = list(map(lambda x: (math.ceil(x),sim_qx(n1Qx(1,math.ceil(x)))),muestra1))
    summ = 0
    for i in simulacion:
        if i[1]:
            summ+=1
    #print("Ganancias: ", prima*TAMAÑO_MUESTRA - summ)
    ganancias.append(prima*TAMAÑO_MUESTRA-summ)
    
    print("iteracion:", I)

siniestralidad_arreglo1 = ganancias
promedio=np.mean(siniestralidad_arreglo1)
mediana=np.median(siniestralidad_arreglo1)
minimo=min(siniestralidad_arreglo1)
maximo=max(siniestralidad_arreglo1)
desv_est=np.std(siniestralidad_arreglo1)
print("promedio", promedio)
print("mediana", mediana)
print("minimo", minimo)
print("maximo", maximo)
print("desv_est", desv_est)
    

# 2