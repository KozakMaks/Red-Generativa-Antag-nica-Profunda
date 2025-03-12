# Proyecto DCGAN con PyTorch

## Resumen del Proyecto

Este proyecto implementa una Red Generativa Antagónica Profunda (DCGAN) utilizando PyTorch, cuyo objetivo es generar imágenes realistas inspiradas en el dataset CIFAR-10. La arquitectura DCGAN aprovecha dos redes neuronales en competencia: el generador, que crea imágenes a partir de vectores de ruido, y el discriminador, que evalúa la autenticidad de dichas imágenes. El entrenamiento conjunto de ambos modelos permite que el generador produzca imágenes cada vez más convincentes.

## Características Principales

- **Arquitectura DCGAN:** Implementa tanto el generador como el discriminador con múltiples capas convolucionales, normalización por lotes y funciones de activación optimizadas para la generación de imágenes.
- **Entrenamiento con CIFAR-10:** Utiliza el dataset CIFAR-10, compuesto por 60,000 imágenes a color de 32x32 píxeles en 10 clases, para entrenar la red.
- **Interfaz Gráfica:** Una aplicación de escritorio basada en Tkinter que permite generar imágenes a partir del modelo entrenado, visualizarlas en tiempo real y guardarlas en disco.
- **Uso de GPU:** Optimizado para utilizar GPU si está disponible, lo que acelera tanto el entrenamiento como la generación de imágenes.

## Contenido del Repositorio

- **`train.py`**: Script para entrenar el modelo DCGAN. Incluye la definición de la arquitectura, el ciclo de entrenamiento y la persistencia del modelo generador en el archivo `generator.pth`.
- **`app.py`**: Aplicación gráfica que carga el modelo generador entrenado y permite generar nuevas imágenes mediante ruido aleatorio, visualizándolas y ofreciendo la opción de guardarlas.
- **`README.md`**: Este documento, que ofrece una descripción completa del proyecto, instrucciones de instalación, uso, detalles técnicos y más.

## Instalación y Requisitos

Asegúrate de tener instalados los siguientes paquetes y herramientas:

- Python 3.x
- [PyTorch](https://pytorch.org/) y [torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://python-pillow.org/)
- Tkinter (incluido en la mayoría de las distribuciones de Python)

Para instalar las dependencias, ejecuta:

```bash
pip install torch torchvision pillow
