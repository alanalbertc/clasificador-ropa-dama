"""
Autor: Alan Carmona
Fecha: 04/01/23
Versión: 1.0
        
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_sslify import SSLify
from werkzeug.utils import secure_filename
import pandas as pd
from pymongo import MongoClient
import os
import numpy as np
import cv2 as cv
import lightgbm as lgb
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
import otsu
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# CORS(app, resources={r"/*": {"origins": "https://trabajo-terminal-sigma.vercel.app/"}})
# sslify = SSLify(app)

UPLOAD_FOLDER = r'./static/images'
UMBRAL_FOLDER = r'./static/images/umbraladas'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UMBRAL_FOLDER'] = UMBRAL_FOLDER


ESTENSIONES_PERMITIDAS = {'png', 'jpg'}
# Garantiza que una imágen sea del tipo 'ESTENSIONES_PERMITIDAS'
def extension_permitida(archivo):
    return '.' in archivo and archivo.rsplit('.', 1)[1].lower() in ESTENSIONES_PERMITIDAS


def umbralar_imagen(image):
    
    # if(not extension_permitida(imagen_path)):
    #     print('Extensión de imágen no válida')

    # Leer la imagen
    # image = cv.imread(imagen_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Obtención de los valores mínimos y máximos para colores en HSV
    medianBlur = cv.medianBlur(gray, 7)
    umbrales = otsu.otsu_multithreshold(medianBlur)

    hist = cv.calcHist([gray], [0], None, [256], [0, 255])
    seccion1 = hist[:umbrales[0]]
    seccion2 = hist[umbrales[0]:umbrales[1]]
    seccion3 = hist[umbrales[1]:]

    count1 = int(seccion1.sum())
    count2 = int(seccion2.sum())
    count3 = int(seccion3.sum())

    # Crear una lista de tuplas para regiones y su suma de pixeles
    regiones_ordenadas = [('R1', count1), ('R2', count2), ('R3', count3)]
    # Ordenar la lista de acuerdo a la suma de pixeles
    regiones_ordenadas.sort(key=lambda x: x[1], reverse=True)

    # Comparación para encontrar la sección con más píxeles
    seccion_con_mas_pixeles = regiones_ordenadas[0][0]
    if seccion_con_mas_pixeles == 'R1':
        umbralbajo = np.array(0)
        umbralalto = np.array(umbrales[0])
    elif seccion_con_mas_pixeles == 'R2':
        umbralbajo = np.array(umbrales[0])
        umbralalto = np.array(umbrales[1])
    else:
        umbralbajo = np.array(umbrales[1])
        umbralalto = np.array(255)

    th_image = cv.inRange(medianBlur, umbralbajo, umbralalto)

    alto, ancho = th_image.shape
    p_esquina1 = th_image[0, 0]
    p_esquina2 = th_image[0, ancho - 1]
    p_esquina3 = th_image[alto - 1, 0]
    p_esquina4 = th_image[alto - 1, ancho - 1]

    if (p_esquina1 == 255 and p_esquina4 == 255) or (p_esquina2 == 255 and p_esquina3 == 255):
        # Si el fondo es más grande, se vuelve a binarizar pero ahora con la segunda región más grande
        seccion_con_mas_pixeles = regiones_ordenadas[1][0]
        if seccion_con_mas_pixeles == 'R1':
            umbralbajo = np.array(0)
            umbralalto = np.array(umbrales[0])
        elif seccion_con_mas_pixeles == 'R2':
            umbralbajo = np.array(umbrales[0])
            umbralalto = np.array(umbrales[1])
        else:
            umbralbajo = np.array(umbrales[1])
            umbralalto = np.array(255)

        th_image = cv.inRange(medianBlur, umbralbajo, umbralalto)
    
    return th_image


def calcular_caracteristicas_color(img):
    #Convertir la imágen a HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Calculate mean and standard deviation for each channel (H, S, V)
    std_hue = np.std(hsv[:, :, 0])
    std_saturation = np.std(hsv[:, :, 1])
    std_value = np.std(hsv[:, :, 2])

    return std_hue, std_saturation, std_value

def extraer_caracteristicas(img, img_umbralada):
    TAM_PIXELES = 39
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    
    imagen_caracteristicas = pd.DataFrame()
    df = pd.DataFrame()
    
    # Características de color
    std_hue, std_saturation, std_value = calcular_caracteristicas_color(img)
    df['Std_Hue'] = [round(std_hue, 8)]
    df['Std_Saturation'] = [round(std_saturation, 8)]
    df['Std_Value'] = [round(std_value, 8)]

    # GLCM (Gray level co-occurrence matrices)
    GLCM2 = graycomatrix(img_gray, [3], [0])       
    GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
    df['Energy'] = [round(float(GLCM_Energy2), 8)]
    GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
    df['Correlation'] = [round(float(GLCM_corr2), 8)]             
    GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
    df['homogeneity'] = [round(float(GLCM_hom2), 8)]       
    GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
    df['contrast'] = [round(float(GLCM_contr2), 8)]
    img_etiquetada = measure.label(img_umbralada)
    regiones = measure.regionprops(img_etiquetada)
                
    for region in regiones:
        if region.area > TAM_PIXELES:       

            diameter = region.equivalent_diameter_area
            feret = region.feret_diameter_max

            forma = feret / diameter

            df['moment_hu1'] = [round(region.moments_hu[0], 8)]
            df['moment_hu4'] = [round(region.moments_hu[3], 8)]
            df['forma'] = [round(forma, 8)]
            
    # Verificar si el dataframe no está vacío antes de concatenar
    if not df.empty:
        imagen_caracteristicas = pd.concat([imagen_caracteristicas, df])
        
    return imagen_caracteristicas



def distancia_euclidiana(features1, features2):
    # Extraer los valores de las filas como arrays
    array1 = features1.values
    array2 = features2.values

    # Convertir los arrays a tipo float si no lo son
    array1 = array1.astype(float)
    array2 = array2.astype(float)

    # Distancia euclidiana
    distancia = np.sqrt(np.sum((array1 - array2)**2))
    return distancia

import numpy as np

def distancia_manhattan(features1, features2):
    # Extraer los valores de las filas como arrays
    array1 = features1.values
    array2 = features2.values

    # Convertir los arrays a tipo float si no lo son
    array1 = array1.astype(float)
    array2 = array2.astype(float)

    # Distancia de Manhattan
    distancia = np.sum(np.abs(array1 - array2))
    return distancia





import pandas as pd

def buscar_imagen_similar(img_features, collection, class_name):
    # Realizar la consulta MongoDB y almacenar los resultados en una lista
    resultados = list(collection.find({'class': class_name}))
    print(f"len(resultados): {len(resultados)}")    #debug
    

    # Inicializar con un valor alto para encontrar el mínimo
    min_distance = float('inf')
    most_similar_img = None

    # Iterar sobre los resultados almacenados en la lista
    for img_data in resultados:
        # Crear un DataFrame a partir de img_data
        stored_features = pd.DataFrame([img_data])

        # Extraer las columnas relevantes para el cálculo de la distancia
        stored_features = stored_features[["Std_Hue", "Std_Saturation", "Std_Value", "Energy", "Correlation", "homogeneity", "contrast", "moment_hu1", "moment_hu4", "forma"]]

        # similarity = distancia_euclidiana(img_features, stored_features)
        similarity = distancia_euclidiana(img_features, stored_features)

        # Actualizar si encontramos una distancia menor
        if similarity < min_distance:
            min_distance = similarity
            most_similar_img = img_data["nombre"]    


    return most_similar_img

def realizar_prediccion(img):
    SIZE = 128
    
    clases = ['black_dress', 'blue_dress', 'red_dress', 'white_dress']
    
    # Codifcación de nombres de clases a [0, 1, 2, 3]
    le = LabelEncoder()
    le.fit(clases)
    le.transform(clases)
    
    # Leer imágen y reajustar el tamaño       
    # img = cv.imread(ruta_img)
    img = cv.resize(img, (SIZE, SIZE)) 
            
    # modelo = lgb.Booster(model_file='static\modelo.txt')
    modelo = lgb.Booster(model_file='modelo/modelo.txt')
            
    # Extraer características 
    img_umbralada = umbralar_imagen(img)
    # Guardar imagen umbralada
    umbral_filename = f"umbral_0.jpg"
    umbral_file_path = os.path.join(app.config['UMBRAL_FOLDER'], umbral_filename)
    cv.imwrite(umbral_file_path, img_umbralada)
    
    # Ajustar la dimensión a la adecuada para el modelo
    img_umbralada = cv.resize(img_umbralada, (SIZE, SIZE))
    img_umbralada = np.array(img_umbralada)
    img_np = np.expand_dims(img, axis=0)     # Expandir dimensiones
    img = np.array(img)
    caracteristicas_img = extraer_caracteristicas(img, img_umbralada)
    print(f"caracteristicas_img_input.shape: {caracteristicas_img.shape}")
    print(f"caracteristicas_img_input: {caracteristicas_img}")
    caracteristicas_img_np = np.expand_dims(caracteristicas_img, axis=0)
    input_img_for_RF = np.reshape(caracteristicas_img_np, (img_np.shape[0], -1))
            
    #Predict
    print(f"input_img_for_RF.shape: {input_img_for_RF.shape}") 
    class_prediction = modelo.predict(input_img_for_RF) #Utilizando el modelo cargado!!
    print(f"prediction shape: {type(class_prediction.shape)}")
    class_prediction = np.argmax(class_prediction, axis=1)
    # class_number = int(class_prediction)
    class_prediction = str(le.inverse_transform(class_prediction.ravel())[0])  #Reverse the label encoder to original name
    print("The prediction for this image is: ", class_prediction)
    
    # Conectar a MongoDB
    # Cambiar campos <conection_string> por la cadena de conexión a MongoDB
    try:
    
        client = MongoClient('mongodb+srv://admin:kE_ODa085Kro%5CA@clasificador-cluster.jjkxafo.mongodb.net/')
        db = client['clothes']
        collection = db['features']
        
        print('Conexión exitosa a MongoDB')
        
        print(f'size(collection): {collection.find().count()}')
    
    except Exception as e:
        print(f'Error al conectar a MongoDB {e}')

    # Encontrar a la imágen más similar dentro de la clase 
    result_image = buscar_imagen_similar(caracteristicas_img, collection, class_prediction)
    print(f'IMAGEN RESULTADO: {result_image}')
    
    return class_prediction, result_image


def process_image(file):
    # Leer la imagen como datos binarios
    image_data = file.read()

    # Decodificar los datos binarios y convertirlos en una matriz NumPy
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)


    return img  # Puedes devolver la imagen procesada si es necesario



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and extension_permitida(file.filename):
            # filename = secure_filename(file.filename)
            # file_path = app.config['UPLOAD_FOLDER'] + '/' + filename
            # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # file.save(file_path)
            
            # Procesar la imagen sin guardarla físicamente en el sistema de archivos
            processed_image = process_image(file)
            
            class_prediction, img_similar = realizar_prediccion(processed_image)

            # return render_template('index.html', result_image=result_image)
            # return render_template('index.html', input_image=filename, class_prediction=class_prediction, img_similar=img_similar)
            
            # La ruta del servidor se establece desde el front
            # ruta_servidor = 'http://20.241.183.28:5000'
            # ruta_servidor = 'http://127.0.0.1:5000'
            
            ruta_front = 'static/dataset/' + class_prediction + '/'+ img_similar
            return jsonify({'img_similar': ruta_front})
            # return jsonify({'error': 'No selected file'})
            # get_img(): Regresar ruta completa (ya construida)
                # Formato json con atributo ruta {nombre_atributo_ruta: 'asdfsd/sdfsd'}
            # Cross-origin: checar si deja acceder al dataset
                # Hay que avisarle al backend que se va a acceder desde fuera
                # Lista blanca de los lugares
            # Post-man: regresar imágen
            # fetch: construir la ruta y mandarla
            #  

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)  #Se define 0.0.0.0 para docker