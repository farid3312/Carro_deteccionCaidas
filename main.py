# main.py
import asyncio
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import web
from services.esp32_ws import conectar_esp32_ws

# Lifespan reemplaza a los eventos obsoletos startup/shutdown en FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicia la conexión a la ESP32 en segundo plano
    tarea_esp32 = asyncio.create_task(conectar_esp32_ws())
    yield
    # Limpieza al apagar el servidor
    tarea_esp32.cancel()
    try:
        await tarea_esp32
    except asyncio.CancelledError:
        print("\nServidor cerrado correctamente.")

app = FastAPI(title="Panel de Control ESP32 y YOLOv8", lifespan=lifespan)

# Esta es la línea que habilita la lectura de tu carpeta "data"
app.mount("/data", StaticFiles(directory="data"), name="data")

# Incluir las rutas
app.include_router(web.router)