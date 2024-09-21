import json

from flask import Flask, request, render_template, redirect, url_for, jsonify
import time
import cv2

from PIL import Image
import requests
from io import BytesIO
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from flask_lt import run_with_lt
app = Flask(__name__)
run_with_lt(app)

@app.route('/')
def home():
    return render_template('home.html')

def filtro_gaussiano(tamaño, sigma):
    filtro = [[0 for _ in range(tamaño)] for _ in range(tamaño)]
    suma = 0
    medio = tamaño // 2

    for x in range(-medio, medio+1):
        for y in range(-medio, medio+1):
            valor = (1 / (2 * 3.1416 * sigma**2)) * 2.71828**(-(x**2 + y**2) / (2 * sigma**2))
            filtro[x+medio][y+medio] = valor
            suma += valor

    # Normalizar el filtro
    for i in range(tamaño):
        for j in range(tamaño):
            filtro[i][j] /= suma

    return filtro

def aplicar_filtro_pycuda(imagen, filtro):
    # Inicializa el dispositivo CUDA
    cuda.init()
    device = cuda.Device(0)  # Asume que estás usando el primer dispositivo CUDA
    context = device.make_context()

    alto, ancho, canales = imagen.shape
    tamaño = len(filtro)
    medio = tamaño // 2
    nueva_imagen = np.zeros((alto, ancho, canales), dtype=np.uint8)
    mod = SourceModule("""
    __global__ void aplicar_filtro(float *dest, float *img, float *filtro, int ancho, int alto, int canales, int tamaño)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        if (idx >= ancho || idy >= alto) return;
        for (int c = 0; c < canales; c++) {
            float suma = 0;
            for (int x = -tamaño/2; x <= tamaño/2; x++) {
                for (int y = -tamaño/2; y <= tamaño/2; y++) {
                    int i = min(max(idy+y, 0), alto-1);
                    int j = min(max(idx+x, 0), ancho-1);
                    suma += img[(i*ancho + j)*canales + c] * filtro[(y+tamaño/2)*tamaño + (x+tamaño/2)];
                }
            }
            dest[(idy*ancho + idx)*canales + c] = round(suma);
        }
    }
    """)

    # Obtener la función del kernel de CUDA
    aplicar_filtro = mod.get_function("aplicar_filtro")

    # Convertir la imagen y el filtro a matrices de un solo canal
    imagen_1c = np.float32(imagen.flatten())
    filtro_np = np.array(filtro)  # Convertir el filtro a una matriz de NumPy
    filtro_1c = np.float32(filtro_np.flatten())

    # Crear la matriz de destino
    dest = np.zeros_like(imagen_1c)

    # Aplicar el filtro con CUDA
    aplicar_filtro(cuda.Out(dest), cuda.In(imagen_1c), cuda.In(filtro_1c), np.int32(ancho), np.int32(alto), np.int32(canales), np.int32(tamaño), block=(32, 32, 1), grid=(ancho//32, alto//32))

    # Reshape la matriz de destino a la forma original de la imagen
    nueva_imagen = np.reshape(np.uint8(dest), (alto, ancho, canales))

    # Libera el contexto CUDA
    context.pop()

    return nueva_imagen
def ruido_sal_pimienta_pycuda(imagen, prob, tamaño_region):
    cuda.init()
    device = cuda.Device(0)  # Asume que estás usando el primer dispositivo CUDA
    context = device.make_context()
    alto, ancho, canales = imagen.shape

    # Crear la matriz de destino
    dest = np.zeros_like(imagen)

    # Definir el kernel de CUDA
    mod = SourceModule("""
    __global__ void ruido_sal_pimienta(float *dest, float *img, float *ruido, float prob, int ancho, int alto, int canales, int tamaño_region)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx >= ancho || idy >= alto) return;
        for (int c = 0; c < canales; c++) {
            for (int i = idx; i < min(idx + tamaño_region, ancho); i++) {
                for (int j = idy; j < min(idy + tamaño_region, alto); j++) {
                    float r = ruido[(j*ancho + i)*canales + c];
                    if (r < prob) {
                        dest[(j*ancho + i)*canales + c] = (r < prob/2) ? 0 : 255;
                    } else {
                        dest[(j*ancho + i)*canales + c] = img[(j*ancho + i)*canales + c];
                    }
                }
            }
        }
    }
    """)

    # Obtener la función del kernel de CUDA
    ruido_sal_pimienta = mod.get_function("ruido_sal_pimienta")

    # Generar los números aleatorios en Python
    ruido = np.random.rand(alto, ancho, canales).astype(np.float32)

    # Convertir la imagen a una matriz de un solo canal
    imagen_1c = np.float32(imagen.flatten())
    ruido_1c = ruido.flatten()

    # Crear la matriz de destino
    dest = np.zeros_like(imagen_1c)

    # Aplicar el ruido de sal y pimienta con CUDA
    ruido_sal_pimienta(cuda.Out(dest), cuda.In(imagen_1c), cuda.In(ruido_1c), np.float32(prob), np.int32(ancho), np.int32(alto), np.int32(canales), np.int32(tamaño_region), block=(32, 32, 1), grid=(ancho//32, alto//32))

    # Reshape la matriz de destino a la forma original de la imagen
    nueva_imagen = np.reshape(np.uint8(dest), (alto, ancho, canales))
    # Libera el contexto CUDA
    context.pop()
    return nueva_imagen

# Kernel Sobel
sobel_kernel = """
// Define las matrices de convolución
__constant__ int sobel_filter_x[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__constant__ int sobel_filter_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

__device__ unsigned char get_pixel_value(const unsigned char* image_data, int width, int height, int x, int y, int channel) {
    x = min(max(x, 0), width - 1);
    y = min(max(y, 0), height - 1);
    return image_data[(y * width + x) * channel + channel];
}

__global__ void sobel_filter_kernel(const unsigned char* image_data, int width, int height, int channels, unsigned char* output_image_data, int filter_size, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            int gx = 0;
            int gy = 0;
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    gx += sobel_filter_x[i][j] * get_pixel_value(image_data, width, height, x - filter_size / 2 + j, y - filter_size / 2 + i, channels);
                    gy += sobel_filter_y[i][j] * get_pixel_value(image_data, width, height, x - filter_size / 2 + j, y - filter_size / 2 + i, channels);
                }
            }
            int gradient_magnitude = sqrtf(gx * gx + gy * gy);
            output_image_data[(y * width + x) * channels + c] = (gradient_magnitude > threshold) ? gradient_magnitude : 0;
        }
    }
}
"""
def apply_sobel_filter_GPU_with_variable_threshold(image_data, width, height, channels, filter_size, thresholds):
    cuda.init()
    device = cuda.Device(0)  # Asume que estás usando el primer dispositivo CUDA
    context = device.make_context()
    mod = SourceModule(sobel_kernel)

    sobel_filter_kernel = mod.get_function("sobel_filter_kernel")

    d_image_data = cuda.mem_alloc(image_data.nbytes)
    cuda.memcpy_htod(d_image_data, image_data)

    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    output_image_data = cuda.mem_alloc(image_data.nbytes)

    sobel_filter_kernel(d_image_data, np.int32(width), np.int32(height), np.int32(channels), output_image_data, np.int32(filter_size), np.int32(thresholds[filter_size]),
                        block=block_size, grid=grid_size)

    output_host = np.empty_like(image_data)
    cuda.memcpy_dtoh(output_host, output_image_data)
    context.pop()
    return output_host
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    tamaño = int(request.form['tamaño'])
    opcion = request.form['opcion']
    if file:
        if opcion == '1':
            img = Image.open(file)
            imagen_np = np.array(img)
            img.save('static/images/uploaded_image.jpg')
            start_time = time.time()
            filtro = filtro_gaussiano(tamaño, 8)
            imagen_filtrada = aplicar_filtro_pycuda(imagen_np, filtro)
            end_time = time.time()
            tiempo_ejecucion = end_time - start_time
            Image.fromarray(imagen_filtrada).save('static/images/Resultado.jpg')
            # Guarda el tiempo de ejecución en un archivo JSON
            with open('static/tiempo_ejecucion.json', 'w') as f:
                json.dump({'tiempo_ejecucion': tiempo_ejecucion}, f)
            return redirect(url_for('show_image'))
        elif opcion == '2':
            img = Image.open(file)
            imagen_np = np.array(img)
            img.save('static/images/uploaded_image.jpg')
            start_time = time.time()
            imagen_ruidosa = ruido_sal_pimienta_pycuda(imagen_np, 0.1, tamaño)

            end_time = time.time()
            tiempo_ejecucion = end_time - start_time
            Image.fromarray(imagen_ruidosa).save('static/images/Resultado.jpg')
            # Guarda el tiempo de ejecución en un archivo JSON
            with open('static/tiempo_ejecucion.json', 'w') as f:
                json.dump({'tiempo_ejecucion': tiempo_ejecucion}, f)
            return redirect(url_for('show_image'))
        elif opcion == '3':
            img = Image.open(file)
            thresholds = {9: 1500, 13: 1300, 21: 1700}
            img.save('static/images/uploaded_image.jpg')
            image = cv2.imread('static/images/uploaded_image.jpg')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            start_time = time.time()

            filtered_image = apply_sobel_filter_GPU_with_variable_threshold(gray_image, gray_image.shape[1],
                                                                            gray_image.shape[0], 1, tamaño, thresholds)
            end_time = time.time()
            tiempo_ejecucion = end_time - start_time
            Image.fromarray(filtered_image).save('static/images/Resultado.jpg')
            # Guarda el tiempo de ejecución en un archivo JSON
            with open('static/tiempo_ejecucion.json', 'w') as f:
                json.dump({'tiempo_ejecucion': tiempo_ejecucion}, f)
            return redirect(url_for('show_image'))




@app.route('/upload_url', methods=['POST'])
def upload_url():
    url = request.form['url']
    tamaño = int(request.form['tamaño'])
    opcion = request.form['opcion']
    if url:
        if opcion == '1':
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save('static/images/uploaded_image.jpg')
            imagen_np = np.array(img)
            start_time = time.time()
            filtro = filtro_gaussiano(tamaño, 8)
            imagen_filtrada = aplicar_filtro_pycuda(imagen_np, filtro)
            end_time = time.time()
            tiempo_ejecucion = end_time - start_time
            Image.fromarray(imagen_filtrada).save('static/images/Resultado.jpg')
            # Guarda el tiempo de ejecución en un archivo JSON
            with open('static/tiempo_ejecucion.json', 'w') as f:
                json.dump({'tiempo_ejecucion': tiempo_ejecucion}, f)
            return redirect(url_for('show_image'))
        elif opcion == '2':
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save('static/images/uploaded_image.jpg')
            imagen_np = np.array(img)
            start_time = time.time()
            imagen_ruidosa = ruido_sal_pimienta_pycuda(imagen_np, 0.1, tamaño)
            end_time = time.time()
            tiempo_ejecucion = end_time - start_time
            Image.fromarray(imagen_ruidosa).save('static/images/Resultado.jpg')
            # Guarda el tiempo de ejecución en un archivo JSON
            with open('static/tiempo_ejecucion.json', 'w') as f:
                json.dump({'tiempo_ejecucion': tiempo_ejecucion}, f)
            return redirect(url_for('show_image'))
        elif opcion == '3':
            thresholds = {9: 1500, 13: 1300, 21: 1700}
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save('static/images/uploaded_image.jpg')
            image = cv2.imread('static/images/uploaded_image.jpg')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            start_time = time.time()

            filtered_image = apply_sobel_filter_GPU_with_variable_threshold(gray_image, gray_image.shape[1],
                                                                            gray_image.shape[0], 1, tamaño, thresholds)
            end_time = time.time()
            tiempo_ejecucion = end_time - start_time
            Image.fromarray(filtered_image).save('static/images/Resultado.jpg')
            # Guarda el tiempo de ejecución en un archivo JSON
            with open('static/tiempo_ejecucion.json', 'w') as f:
                json.dump({'tiempo_ejecucion': tiempo_ejecucion}, f)
            return redirect(url_for('show_image'))



@app.route('/show_image')
def show_image():
    return render_template('show_image.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
