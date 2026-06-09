# 🚗 Robótica Edge & IA: Sistema Desacoplado de Telemetría y Detección de Caídas en Tiempo Real

Este proyecto es un ecosistema completo de hardware y software diseñado para la adquisición de telemetría inercial y el monitoreo robótico mediante visión computacional. Su objetivo principal es la **detección autónoma de caídas humanas en tiempo real** utilizando algoritmos de Deep Learning.

Para superar las limitaciones de hardware de los microcontroladores convencionales, el sistema implementa una **Arquitectura de Pipeline Desacoplado**:
1. **Edge (ESP32-S3-CAM):** Actúa puramente como un nodo de adquisición de alta velocidad, capturando video raw, datos de proximidad (HC-SR04) y matrices inerciales (GY-6500).
2. **Core (FastAPI & YOLOv8):** Un servidor central recibe el flujo multiplexado por WebSockets y delega el análisis espacial a una GPU (aceleración CUDA), manteniendo tiempos de inferencia por debajo de los 30 ms.
3. **Frontend (Web Canvas):** Renderiza el streaming de video fluido a ~30 FPS y dibuja dinámicamente las coordenadas de detección (STANDING, SITTING, FALL) sin retrasar el video.

### ✨ Características Principales
* **Baja Latencia (Zero-Lag):** Gestión de buffers duales (OPI PSRAM) y abandono dinámico de frames saturados (`canSend`).
* **Telemetría Multiplexada:** Transmisión simultánea de video binario y datos inerciales por I2C a 400kHz.
* **Control HTTP Persistente:** Manejo del chasis robótico (L293D) mitigando el retraso del Handshake TCP tradicional.
* **Filtro contra Falsos Positivos:** Validación temporal de 3 fotogramas consecutivos para la emisión de alertas críticas.
* **Frenado Autónomo de Emergencia:** Decisión de corte de tracción a nivel de hardware nativo para evitar colisiones (≤ 5cm).

Arquitectura:
<img width="3000" height="2743" alt="image" src="https://github.com/user-attachments/assets/c972f8e1-140f-43fe-8164-68fbfdb7e92a" />


Configuración Crítica del Compilador de Hardware (Arduino IDE)
Para alcanzar el rendimiento de transferencia documentado, la placa ESP32-S3 debe compilarse con los siguientes flags de optimización del silicio:

Frecuencia de CPU: 240MHz (WiFi)

Flash Mode: QIO 80MHz

PSRAM: OPI PSRAM (Requisito fundamental para dar soporte a los buffers duales asignados a la captura de descriptores JPEG directos de la cámara).

### A. Canal WebSocket Multiplexado (Telemetría de Entrada)
Para evitar el costo de empaquetado y desempaquetado de protocolos pesados, la ESP32 transmite datos estructurados de forma binaria o textual a través del endpoint remoto `ws://<IP_FIJA>/ws`:

* **Transmisión de Video Crudo (Datos Binarios):** Cada 33 milisegundos (~30 FPS), la cámara extrae el búfer de imagen en formato JPEG (`bytes`). El servidor FastAPI clona el búfer y realiza un bypass directo hacia la interfaz gráfica mediante el WebSocket exclusivo `/ws_video`. Esto garantiza una fluidez nativa y constante.
* **Transmisión de Telemetría Inercial y de Proximidad (Datos de Texto):** De forma desfasada en tiempo para mitigar colisiones en el búfer de red, la placa envía strings serializados:
  * El sensor de ultrasonido **HC-SR04** transmite distancias en formato `"US:<distancia_cm>"` cada 100ms.
  * El sensor inercial **GY-6500** transmite las matrices de aceleración y giroscopio (`Ax, Ay, Az, Gx, Gy, Gz`) cada 105ms sobre un bus I2C a **400 kHz**.
  El servidor FastAPI intercepta las cadenas de texto y las retransmite al navegador mediante el WebSocket secundario `/ws_datos`.

### B. Canal HTTP Persistente (Comandos de Salida / Actuación)
El control de la tracción diferencial del chasis (Puente H L293D) y la configuración de registros internos del sensor de imagen de la cámara se gestionan mediante el protocolo HTTP de petición-respuesta.

Para eliminar la latencia crítica asociada al *Handshake TCP* repetitivo en cada pulsación del usuario, el servidor FastAPI inicializa una instancia de conexión única y permanente utilizando un cliente persistente de alto rendimiento (`httpx.AsyncClient`). Cuando el navegador dispara eventos táctiles o clic mediante `fetch('/car?d=F')`, el proxy asíncrono redirige el comando inmediatamente a la velocidad de la red local, estabilizando los tiempos de respuesta del actuador por debajo de los 10ms.

---

## 3. Arquitectura del Lado del Servidor (FastAPI & IA)

El cuello de botella computacional clásico en visión artificial se erradica mediante dos técnicas de optimización multihilo:

### A. Desacoplamiento de la Inferencia YOLOv8
Las funciones de decodificación matricial y trazado de OpenCV son sincrónicas y bloqueantes por naturaleza. Si el modelo YOLOv8 procesara el fotograma dentro del bucle de red (Event Loop), detendría el flujo asíncrono durante el tiempo que dure la inferencia. 

Para evitar esto, el sistema extrae el último búfer disponible de la memoria global (`state.ultimo_frame_bytes`) y delega la ejecución computacional a un hilo secundario del procesador mediante el método `asyncio.to_thread()`. De este modo, la descarga del video desde la ESP32 nunca se interrumpe.

### B. Optimización de Inferencia y Filtros Post-Procesamiento
* **Aceleración por Hardware (CUDA):** El modelo matemático YOLOv8 se transfiere por completo a la memoria VRAM de la GPU integrada (**NVIDIA GTX 1650**) mediante el entorno de ejecución de CUDA. El tiempo de inferencia decae de ~120ms (CPU) a **~30ms (GPU)** por frame a una resolución optimizada de cuadrícula fija de 320 píxeles (`imgsz=320`).
* **Filtro de Historial Temporal contra Falsos Positivos:** Para evitar disparos accidentales de alertas provocados por oclusiones rápidas de la cámara o ruidos ópticos, la lógica del servidor exige una validación ininterrumpida de **3 fotogramas consecutivos** con la etiqueta `FALL` detectada a nivel de clasificación de objetos. Si el hilo de la IA cuenta 3 frames sucesivos estables, se emite el flag estructurado de alarma `alerta_ia` hacia el frontend.

---

## 4. Arquitectura del Lado de la Interfaz (Frontend Web)

El frontend asume la responsabilidad total de empalmar la interfaz gráfica dinámica, eliminando la sobrecarga del servidor en el dibujado de imágenes fijas.

### A. Transformación Matricial de Coordenadas de Bounding Boxes
Dado que la ESP32 transmite el video a una resolución QVGA nativa (320x240) para maximizar la velocidad de transferencia WiFi, pero la pantalla renderiza la imagen a un tamaño dinámico escalado por CSS, los píxeles absolutos entregados por YOLOv8 no coinciden directamente con la pantalla del usuario.

El cliente soluciona esto calculando dinámicamente factores de escala en tiempo real en cada fotograma recibido:

$$\text{scale}_X = \frac{\text{canvas.width}}{\text{resoluciónOriginal.w}}$$

$$\text{scale}_Y = \frac{\text{canvas.height}}{\text{resoluciónOriginal.h}}$$

Donde las coordenadas proyectadas finales sobre el elemento vectorial transparentado se determinan mediante:

$$X_{\text{pantalla}} = X_{\text{YOLO}} \times \text{scale}_X$$

$$Y_{\text{pantalla}} = Y_{\text{YOLO}} \times \text{scale}_Y$$

### B. Gestión de Memoria en el Navegador
Para prevenir fugas de memoria RAM e inanición de recursos en el navegador provocadas por el ciclo intensivo de actualización de imágenes a 30 FPS, la inyección binaria del objeto `Blob` se enlaza al manejador de eventos de carga estructural de la imagen (`imgElement.onload`), destruyendo y liberando las referencias de memoria obsoletas del recolector mediante la instrucción explícita `URL.revokeObjectURL(url)` inmediatamente después de su visualización física en la pantalla.

### C. Clasificación Visual de Posturas
El script inyecta contextos gráficos diferenciados por color sobre el `<canvas>` según la etiqueta normalizada extraída en mayúsculas:
* `FALL`: Recuadro de alta visibilidad rojo intenso (`#ff1744`) con parpadeo de advertencia en el contenedor superior del visor.
* `SITTING`: Recuadro ámbar/naranja de advertencia pasiva (`#ff9100`) con inversión tipográfica para legibilidad de contraste.
* `STANDING`: Recuadro azul cobalto de estado operacional normal (`#00b0ff`).

---

## 5. Especificaciones de Configuración y Despliegue Técnico

### Requisitos del Sistema Operativo y Dependencias
Consulte las dependencias completas del entorno en el archivo adjunto `requirements.txt`. El software base requiere Python instalado en versiones $\geq 3.10$.

Para la correcta inicialización de la GPU NVIDIA local, es mandatorio asegurar la presencia de los controladores del compilador CUDA versión 11.8 o superior, acompañados de los binarios distribuidos de PyTorch con soporte de aceleración nativa:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```
Resumen Final :

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/b564cd1b-2614-4065-bd1a-84a465744767" />

I. Requisitos de Hardware (Vehículo Autónomo / Edge)ESP32-S3-CAM:

Microcontrolador principal. Debe tener soporte para memoria OPI PSRAM habilitado para manejar los búferes de video duales.

Módulo GY-6500 (MPU6500): Sensor inercial de 6 ejes (Giroscopio y Acelerómetro) para la telemetría, conectado vía I2C.

Módulo HC-SR04: Sensor ultrasónico para el sistema de evasión de colisiones y freno autónomo de emergencia.

Driver L293D (Puente H): Controlador para la tracción de los motores.

Chasis y Motores DC: Base robótica móvil, incluyendo mínimo dos motores DC con reductores y sus respectivas ruedas.

Fuente de Alimentación: Batería LiPo (ej. 2S 7.4V) para alimentar los motores a través del L293D, combinada con un módulo reductor de voltaje (Buck Converter) configurado a 5V para alimentar la ESP32 de forma estable.

Cables y Ensamblaje: Protoboard o placa PCB pre-perforada, cables jumper (macho-macho, macho-hembra).

II. Requisitos de Hardware

(Estación Base / Servidor)Procesador (CPU): Intel Core i5 / AMD Ryzen 5 o superior.

Tarjeta Gráfica (GPU): Tarjeta gráfica dedicada NVIDIA con soporte nativo para arquitectura CUDA (ej. GTX 1650, RTX serie 3000/4000).Es obligatoria para mantener el tiempo de inferencia de YOLOv8 por debajo de los 30ms.

Red Local: Un enrutador Wi-Fi (idealmente de banda 5GHz o 2.4GHz descongestionada) para alojar a la ESP32 y al servidor en la misma subred estática (192.168.20.x).

III. Requisitos de Software y DependenciasEntorno de Microcontrolador: Arduino IDE con el paquete de placas esp32 de Espressif instalado.

Entorno Servidor: Python $\geq$ 3.10.

Dependencias de Python (Servidor):fastapiuvicornwebsocketshttpxjinja2ultralytics (YOLOv8)numpyopencv-pythonMotor Matemático GPU: PyTorch configurado específicamente para compilar con la tarjeta gráfica:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118






