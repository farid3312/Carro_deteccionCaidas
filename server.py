import asyncio
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import websockets
from ultralytics import YOLO

# 1. Inicializar la aplicación y cargar el modelo
app = FastAPI(title="Servidor YOLOv8 y ESP32 WebSocket")
ruta_modelo = "models/yolov8n/best.pt"
print(f"Cargando modelo desde: {ruta_modelo}")
model = YOLO(ruta_modelo)
NOMBRES_CLASES = model.names

# Variable global para almacenar el último fotograma procesado
ultimo_frame_bytes = None

async def conectar_esp32_ws():
    """Conecta al WebSocket, recibe datos, procesa con YOLO y actualiza el fotograma global."""
    global ultimo_frame_bytes
    # La IP fija extraída de tu archivo ayuda.txt
    uri = "ws://192.168.20.200/ws" 
    
    while True:
        try:
            print(f"Conectando a {uri}...")
            async with websockets.connect(uri) as websocket:
                print("Conectado exitosamente al WebSocket de la ESP32.")
                
                async for mensaje in websocket:
                    # Diferenciar si el mensaje es una imagen (bytes) o texto (sensores)
                    if isinstance(mensaje, bytes):
                        # 1. Convertir los bytes crudos a un array de numpy
                        nparr = np.frombuffer(mensaje, np.uint8)
                        # 2. Decodificar el array a una imagen manejable por OpenCV
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            continue
                        
                        # 3. Inferencia con YOLOv8
                        resultados = model(frame, stream=True)
                        for resultado in resultados:
                            # Dibujar las cajas
                            frame = resultado.plot()
                            
                            # Analizar las detecciones buscando caídas
                            for caja in resultado.boxes:
                                clase_id = int(caja.cls[0])
                                nombre_clase = NOMBRES_CLASES[clase_id]
                                
                                # LÓGICA CRÍTICA DE ALERTA
                                if nombre_clase.lower() == "fall":
                                    print("\n[ALERTA CRÍTICA] ¡Caída detectada!\n")
                        
                        # 4. Codificar el fotograma procesado para enviarlo al navegador HTTP
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            ultimo_frame_bytes = buffer.tobytes()
                    else:
                        # Si llega texto (ej. los datos del sensor ultrasónico o GY-6500)
                        # Lo imprimimos en consola o lo ignoramos.
                        pass
                        
        except Exception as e:
            print(f"Error de conexión con la ESP32: {e}. Reintentando en 3 segundos...")
            await asyncio.sleep(3)

# 3. Iniciar la conexión al WebSocket al arrancar el servidor
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(conectar_esp32_ws())

async def generar_video():
    """Generador que toma el último frame procesado para servirlo por HTTP."""
    global ultimo_frame_bytes
    while True:
        if ultimo_frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + ultimo_frame_bytes + b'\r\n')
        # Pausa para no saturar el bucle asíncrono (~20 fps)
        await asyncio.sleep(0.05) 

# 4. Endpoints
@app.get("/video")
async def video_feed():
    """Ruta para ver el video procesado en tiempo real desde el navegador."""
    return StreamingResponse(generar_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def inicio():
    return {"mensaje": "Servidor activo. Ve a /video para ver el stream procesado."}