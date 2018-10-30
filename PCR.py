import numpy as np
import matplotlib.pyplot as plt

datos = np.genfromtxt("WDBC.dat",None,delimiter="\n")

#Creacion vector datos diagnosis
vectorDiagnosis = np.zeros(len(datos))

#Creacion vector datos pruebas
matrizDatosPruebas = np.zeros([len(datos),30])
for i in range(len(datos)):
    datosFila = (datos[i].decode('UTF-8')).split(",")
    
    #Clasificacion datos diagnosis
    if datosFila[1]=='M':
        vectorDiagnosis[i]=1
    elif datosFila[1]=='B':
        vectorDiagnosis[i]=0
    
    #Clasificacion datos pruebas
    for j in range(30):
        matrizDatosPruebas[i][j]=((datosFila)[j+2])

#Creacion matriz datos
MatrizDatos = np.transpose(matrizDatosPruebas)

#Normalizacion
for i in range(len(MatrizDatos)):
    MatrizDatos[i]=(MatrizDatos[i]-np.mean(MatrizDatos[i]))/np.std(MatrizDatos[i])

#Definicion funcion covarianza entre dos tabla de datos
def covarianza(datos1,datos2):
    prom1 = np.mean(datos1)
    prom2 = np.mean(datos2)
    return np.sum((datos1-prom1)*(datos2-prom2))/(len(datos1)-1)

#Construccion de la matriz de covarianza de los 31 datos (Diagnosis + 30 pruebas)
matrizCovarizanza = np.empty([len(MatrizDatos),len(MatrizDatos)])
for i in range(len(MatrizDatos)):
    for j in range(len(MatrizDatos)):
        matrizCovarizanza[i,j]=covarianza(MatrizDatos[j,:],MatrizDatos[i,:])
print("Matriz de covarianza:")
print(matrizCovarizanza)

eig = np.linalg.eig(matrizCovarizanza)
autoValores = eig[0]
autoVectores = eig[1]
print("Autovalores: \n" + str(autoValores))
print("Autovectores: \n" + str(autoVectores))


print("En base a lo obtenido con los autovalores y autovectores, se determina que las dos primeras variables son las mas importantes, pues para su magnitud en los autovalores es mucho mayor comparado contra otras variables subsiguientes.")

#Producto punto de PC1 y PC2
#Datos Benigno
PCA1B,PCA2B=np.dot([autoVectores[0],autoVectores[1]],np.transpose(matrizDatosPruebas[vectorDiagnosis==0]))
#Datos Maligno
PCA1M,PCA2M=np.dot([autoVectores[0],autoVectores[1]],np.transpose(matrizDatosPruebas[vectorDiagnosis==1]))

plt.figure()
plt.scatter(PCA2B,PCA1B,color='b',label="Benigno")
plt.scatter(PCA2M,PCA1M,color='r',label="Maligno")
plt.legend()
plt.ylabel("PCA1")
plt.xlabel("PCA2")
plt.savefig("DuarteJohn_PCA.pdf",bbox_inches="tight")

#Descripcion utilidad PCA
print("Dada la grafica obtenida por el metodo de PCA, se concluye que su uso puede ser de bastante utilidad a la hora de crear un diagnostico para un paciente. Como se evidencia, aquellos pacientes con tumores malignos tienden a tener un valor mas negativo en la variable PC2, mientras que aquellos pacientes con diagnosticos benignos tienden a ubicarse al lado derecho.")

