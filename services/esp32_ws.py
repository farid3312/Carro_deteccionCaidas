import asyncio
import cv2
import numpy as np
import json
import websockets
import httpx
from ultralytics import YOLO
from core import state

ruta_modelo = "models/yolov8n/best.pt"
model = YOLO(ruta_modelo)
model.to('cuda')
NOMBRES_CLASES = model.names
URI_ESP32_WS = "ws://192.168.20.200/ws"

async def enviar_datos_al_navegador(mensaje_dict):
    if not state.clientes_web:
        return
    mensaje_json = json.dumps(mensaje_dict)
    for cliente in state.clientes_web.copy():
        try:
            await cliente.send_text(mensaje_json)
        except Exception:
            state.clientes_web.remove(cliente)

async def transmitir_video_crudo(frame_bytes):
    """Envía el frame binario crudo directo al navegador (Cero lag)."""
    if not state.clientes_video:
        return
    for cliente in state.clientes_video.copy():
        try:
            await cliente.send_bytes(frame_bytes)
        except Exception:
            state.clientes_video.remove(cliente)

async def notificar_esp32_caida():
    try:
        async with httpx.AsyncClient() as client:
            await client.get("http://192.168.20.200/car?d=S")
    except Exception:
        pass

def procesar_inferencia(frame_bytes):
    """Se ejecuta en hilo paralelo. Retorna las coordenadas de las cajas y si hay caída."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, False

    # Extraer la resolución original de la imagen que llegó de la ESP32
    alto, ancho = frame.shape[:2] 
    
    resultados = model(frame, imgsz=320, stream=True)
    caida_detectada = False
    lista_cajas = []

    for resultado in resultados:
        for caja in resultado.boxes:
            nombre_clase = NOMBRES_CLASES[int(caja.cls[0])]
            confianza = float(caja.conf[0])
            
            if nombre_clase.lower() == "fall":
                caida_detectada = True
            
            # Extraer coordenadas de la caja
            x1, y1, x2, y2 = caja.xyxy[0].tolist()
            
            lista_cajas.append({
                "clase": nombre_clase,
                "confianza": confianza,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

    # Empaquetamos las coordenadas y el tamaño original para enviarlo al navegador
    datos_deteccion = {
        "cajas": lista_cajas,
        "resolucion": {"w": ancho, "h": alto}
    }
    
    return datos_deteccion, caida_detectada

async def loop_ia_paralelo():
    """Bucle infinito que consume el último frame disponible y ejecuta YOLO."""
    consecutivos_caida = 0  # Contador para validación de 3 FPS
    
    try:
        while True:
            if state.ultimo_frame_bytes is not None:
                frame_a_procesar = state.ultimo_frame_bytes
                state.ultimo_frame_bytes = None
                
                datos_deteccion, caida = await asyncio.to_thread(procesar_inferencia, frame_a_procesar)
                
                # Enviar siempre las coordenadas al navegador para dibujar
                if datos_deteccion is not None:
                    await enviar_datos_al_navegador({
                        "tipo": "detecciones", 
                        "datos": datos_deteccion
                    })
                
                # Lógica de verificación de caídas
                if caida:
                    consecutivos_caida += 1
                    if consecutivos_caida >= 3:  # Solo dispara si van 3 frames seguidos
                        # Enviamos un nuevo tipo de evento: "alerta_ia"
                        await enviar_datos_al_navegador({"tipo": "alerta_ia", "mensaje": "⚠️ IA: ¡CAÍDA DETECTADA!"})
                        # Reseteamos para que no envíe cientos de peticiones por segundo
                        consecutivos_caida = 0 
                else:
                    # Si en un frame la persona ya no está cayendo, se reinicia el contador
                    consecutivos_caida = 0
            
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("Hilo de IA detenido.")
        raise

async def conectar_esp32_ws():
    tarea_ia = asyncio.create_task(loop_ia_paralelo())
    try:
        while True:
            try:
                async with websockets.connect(URI_ESP32_WS) as websocket:
                    print("Conectado a la ESP32.")
                    async for mensaje in websocket:
                        if isinstance(mensaje, bytes):
                            await transmitir_video_crudo(mensaje)
                            state.ultimo_frame_bytes = mensaje
                        else:
                            await enviar_datos_al_navegador({"tipo": "sensor", "mensaje": mensaje})
            except Exception as e:
                print(f"Conexión perdida. Reintentando... ({e})")
                await asyncio.sleep(3)
    except asyncio.CancelledError:
        print("Conexión WebSocket detenida.")
        tarea_ia.cancel() # Cancela también la IA
        raise