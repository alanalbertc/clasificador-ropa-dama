"""
Autor: Alan Carmona
Fecha: 04/01/23
Versión: 1.0
        
"""



import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
import os
import seaborn as sns
import pandas as pd
import skimage
from skimage.feature import graycomatrix, graycoprops
from skimage import measure, preprocessing
import lightgbm as lgb
from sklearn import metrics


def calculate_color_features(image):
    #Convertir la imágen a HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Calculate mean and standard deviation for each channel (H, S, V)
    std_hue = np.std(hsv[:, :, 0])
    std_saturation = np.std(hsv[:, :, 1])
    std_value = np.std(hsv[:, :, 2])

    return std_hue, std_saturation, std_value

def feature_extractor(dataset, datasetBin, img_names):
    PORCENTAJE = 0.05
    
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #Iterar sobre cada imágen
        
        df = pd.DataFrame()  # DataFrame temporal para capturar las características para cada loop
        
        #Limpieza del dataframe después de cada loop
        img = dataset[image, :,:,:]
        imgBin = datasetBin[image, :, :]
        
    
        #Características de color
        std_hue, std_saturation, std_value = calculate_color_features(img)
        df['Std_Hue'] = [std_hue]
        df['Std_Saturation'] = [std_saturation]
        df['Std_Value'] = [std_value]

        #GLCM (Gray level co-ocurrence matrices)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        GLCM2 = graycomatrix(img_gray, [3], [0])       
        GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
        df['Energy'] = GLCM_Energy2
        GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
        df['Correlation'] = GLCM_corr2       
        # GLCM_diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
        # df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
        df['homogeneity'] = GLCM_hom2       
        GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
        df['contrast'] = GLCM_contr2
        img_etiquetada = measure.label(imgBin)
        regiones = measure.regionprops(img_etiquetada)

        
        
        area_porcentaje = 39

        for region in regiones:
            if region.area > area_porcentaje:
                
                diameter = region.equivalent_diameter_area
                feret = region.feret_diameter_max

                forma = feret / diameter

                df['moment_hu1'] = [region.moments_hu[0]]
                df['moment_hu4'] = [region.moments_hu[3]]
                df['forma'] = [forma]
        
        #debug: Agrega nombre de clase
        df['nombre'] = [img_names[image]]
        df['class'] = [le.inverse_transform([y_train[image]])[0]]
        
            
        # Verificar si no está vacío antes de concatenar
        if not df.empty:
            image_dataset = pd.concat([image_dataset, df])
        
    return image_dataset

#Tamaño definido para las imágenes
SIZE = 128


#Inicialización de listas vacias
train_images = []
train_names = []
train_labels = [] 
bin_images = [] 
bin_labes = []

#path del conjunto de entrenamiento del dataset
dataset_directory = r"D:\Documents\Trabajo terminal\TT2\Modelo\Datasets\Apparel-Dresses-splitted\train"
classes_names = os.listdir(dataset_directory)

#Iterar sobre las clases
for clase in classes_names:
    class_path = os.path.join(dataset_directory, clase)
    
    #Iterar sobre las imágenes de cada clase
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        train_names.append(img)  # Almacenar el nombre de la clase
        img = cv.imread(img_path) # Lectura de la imágen a color
        img = cv.resize(img, (SIZE, SIZE)) # Reajuste al tamaño definido
        train_images.append(img)    # Almacenar imágenes en una lista
        train_labels.append(clase)  # Almacenar las etiquetas (clase a la que pertenecen)

        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Hacer exactamente lo mismo para cada imágen en el conjunto de entrenamiento
test_images = []
test_labels = []


dataset_test_directory = r"D:\Documents\Trabajo terminal\TT2\Modelo\Datasets\Apparel-Dresses-splitted\test"
classes_test_names = os.listdir(dataset_test_directory)
for directory_path in classes_test_names:
    class_path = os.path.join(dataset_test_directory, directory_path)
    fruit_label = os.path.split(class_path)[-1]
    # print(fruit_label)
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        # print(img)
        img = cv.imread(img_path) #RLeyendo las imágenes a color
        img = cv.resize(img, (SIZE, SIZE)) #Reajustar tamaño
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Almacenar las imágenes binarizadas del conjunto de entrenamiento en una lista
dataset_bin_directory = r"D:\Images\Apparel-Dresses-splitted-Binarizado\train"
classes_bin_names = os.listdir(dataset_bin_directory)
for directory_path in classes_bin_names:
    class_path = os.path.join(dataset_bin_directory, directory_path)
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        # print(img)
        img = cv.imread(img_path,0) # Lectura en escala de grises
        img = cv.resize(img, (SIZE, SIZE)) # Reajustar el tamaño
        bin_images.append(img)
        
bin_images = np.array(bin_images)

bin_test_images = []

# Almacenar las imágenes binarizadas del conjunto de pruebas en una lista
dataset_bin_test_directory = r"D:\Images\Apparel-Dresses-splitted-Binarizado\test"
classes_bin_test_names = os.listdir(dataset_bin_test_directory)
for directory_path in classes_bin_test_names:
    class_path = os.path.join(dataset_bin_test_directory, directory_path)
    
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        img = cv.imread(img_path,0) # Lectura en escala de grises
        img = cv.resize(img, (SIZE, SIZE)) # Reajustar el tamaño
        bin_test_images.append(img)
        
bin_test_images = np.array(bin_test_images)


#Codificar etiquetas de texto (nombre de clases) a enteros
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Dividir los datso en conjuntos de entrenamiento y prueba de manera convencional

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

print(f"x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"bin_images: {bin_images.shape}")

# Normalizar los pixeles entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#Extracción de características del conjunto de entrenamiento
image_features = feature_extractor(x_train, bin_images, train_names)
# Extraer datos categoricos
image_features.pop('class')
image_features.pop('nombre')

# Guardar las características en un archivo csv (opcional)
# Las características se exportan a la BD
# image_features.to_csv('features.csv',index=False)   #Se almacena en la ruta local
# print('Archivo csv generado con éxito!!')



import lightgbm as lgb
 #Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
d_train = lgb.Dataset(image_features, label=y_train)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':100,
              'num_class':4} 


lgb_model = lgb.train(lgbm_params, d_train, 100)


# Predicción sobre el conjunto de test
# Extraer características y cambiar la forma justo como con el conjunto de entrenamiento
test_features = feature_extractor(x_test, bin_images, train_names)
test_features.pop('class')
test_features.pop('nombre')
print(f"test_features.shape: {test_features.shape}")
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Predecir sobre el conunto de test
print(f"test_for_RF.shape: {test_for_RF.shape}")
test_prediction = lgb_model.predict(test_for_RF)
test_prediction=np.argmax(test_prediction, axis=1)
# Inverdir la codificación para obtener la etiqueta original. 
test_prediction = le.inverse_transform(test_prediction)


# Imprimir el accuracy

print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

# Imprimir la matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(7,7))
sns.set(font_scale=0.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax, fmt='d')


