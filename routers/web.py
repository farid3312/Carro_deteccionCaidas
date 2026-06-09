import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from core import state

router = APIRouter()
templates = Jinja2Templates(directory="templates")
URI_ESP32_HTTP = "http://192.168.20.200"

# Cliente persistente: elimina el retraso del Handshake TCP en cada clic
cliente_http = httpx.AsyncClient(timeout=1.0) 

@router.get("/")
def inicio(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@router.websocket("/ws_video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    state.clientes_video.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.clientes_video:
            state.clientes_video.remove(websocket)

@router.websocket("/ws_datos")
async def websocket_navegador(websocket: WebSocket):
    await websocket.accept()
    state.clientes_web.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.clientes_web:
            state.clientes_web.remove(websocket)

# --- RUTAS PROXY OPTIMIZADAS ---
@router.get("/car")
async def mover_carro(d: str):
    try:
        await cliente_http.get(f"{URI_ESP32_HTTP}/car?d={d}")
    except httpx.RequestError:
        pass
    return {"status": "ok"}

@router.get("/speed")
async def cambiar_velocidad(v: int):
    try:
        await cliente_http.get(f"{URI_ESP32_HTTP}/speed?v={v}")
    except httpx.RequestError:
        pass
    return {"status": "ok"}

@router.get("/enable")
async def habilitar_motor(m: str, v: int):
    try:
        await cliente_http.get(f"{URI_ESP32_HTTP}/enable?m={m}&v={v}")
    except httpx.RequestError:
        pass
    return {"status": "ok"}

@router.get("/control")
async def control_camara(var: str, val: int):
    try:
        await cliente_http.get(f"{URI_ESP32_HTTP}/control?var={var}&val={val}")
    except httpx.RequestError:
        pass
    return {"status": "ok"}