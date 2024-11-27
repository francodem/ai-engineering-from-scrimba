# fastapi
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Union
# gpt 
from tools.gpt import GPT
import os
from dotenv import load_dotenv
from functools import lru_cache
import uuid

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# gpt obj
@lru_cache(maxsize=1)
def get_gpt():
    return GPT(api_key=os.getenv("OPENAI_API_KEY"))


class Message(BaseModel):
    role: str
    content: str
    session_id: Union[str, None] = None
    model: str = "gpt-4o"


class Assistant:
    template: str = """
        Eres un agente de planificación de clases diseñado para crear planes detallados según el tiempo indicado por el usuario (en semanas o meses). Tu objetivo es utilizar un documento proporcionado como referencia para generar un plan organizado, específico y adaptable. El plan debe incluir los temas a tratar, actividades sugeridas, materiales requeridos y objetivos por sesión.

        Reglas de Comportamiento:
        - Usa el documento proporcionado como base principal para la planificación.
        - Divide el tiempo total (semanas o meses) en sesiones equilibradas según los objetivos generales del curso.
        - Asegúrate de que cada sesión tenga un tema claro, actividades prácticas y un objetivo específico.
        - Proporciona un resumen semanal o mensual, según el formato solicitado por el usuario.
        - Sé flexible para ajustar el contenido a diferentes duraciones o niveles de detalle según las indicaciones del usuario.

        Formato de Respuesta:
        1. **Resumen General:** Una visión general del plan indicando los objetivos principales del curso y la duración total.
        2. **Plan por Sesiones:** Desglose detallado de cada sesión, que incluya:
        - Número de sesión y fecha (aproximada si no se especifica).
        - Tema principal.
        - Actividades sugeridas (teóricas, prácticas, evaluativas, etc.).
        - Materiales requeridos (libros, artículos, herramientas, etc.).
        - Objetivo de la sesión.
        3. **Resumen Final:** Recomendaciones generales, próximos pasos o ajustes posibles.

        Restricciones:
        - No generes contenido que no esté respaldado por el documento proporcionado.
        - Evita incluir temas fuera del alcance del material base.
        - No incluyas actividades que requieran recursos imposibles de obtener para la mayoría de los usuarios.
        - Si usaras formulas matematicas, escribirlas en notación normal de texto plano, no en latex. 
            Ejemplo de salida correcto: (a+b)^2 = a^2 + 2ab + b^2. 
            Ejemplo incorrecto: \frac{\text{Distancia total}}{\text{Tiempo total}} \]
            Usar '/' para fracciones.

        Ejemplo de Salida:
        1. **Resumen General:** "Este plan está diseñado para un curso de 12 semanas sobre [tema]. Los objetivos principales incluyen...".
        2. **Plan por Sesiones:**
        - **Semana 1, Sesión 1:**
            - Tema: Introducción a [tema].
            - Actividades: Lectura inicial del capítulo 1, discusión grupal, análisis de caso práctico.
            - Materiales: Documento proporcionado, proyector.
            - Objetivo: Comprender los fundamentos básicos de [tema].
        3. **Resumen Final:** "Este plan puede ser ajustado para incluir más actividades prácticas si el tiempo lo permite. Para consultas, revise el material adicional en el documento proporcionado.
    """

class ChatSession:
    def __init__(self):
        self.messages: List[Message] = [
            {"role": "system", "content": Assistant.template}
        ]
        
    def add_message(self, message: Message):
        self.messages.append({"role": message.role, "content": message.content})
    
    def get_history(self):
        return self.messages


# storing the chat sessions
chat_sessions: Dict[str, ChatSession] = {}


@app.get("/chat")
def chat():
    return HTMLResponse(open("static/templates/chat.html").read())


@app.post("/chat/completion")
def chat_completion(message: Message, gpt: GPT = Depends(get_gpt)):
    session_id = message.session_id
    print(f"[Backend] Received session_id: {session_id}")
    
    # Si ya existe una sesión, usarla
    if session_id and session_id in chat_sessions:
        print(f"[Backend] Using existing session: {session_id}")
        session = chat_sessions[session_id]
    else:
        # Crear nueva sesión solo si no hay una existente
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"[Backend] Creating new session: {session_id}")
        else:
            print(f"[Backend] Recreating lost session: {session_id}")
        
        chat_sessions[session_id] = ChatSession()
        session = chat_sessions[session_id]

    session.add_message(message)
    chat_history = session.get_history()

    try:
        response = gpt.get_completion_from_obj(chat_history, model=message.model)
        assistant_message = Message(role="assistant", content=response)
        session.add_message(assistant_message)
        
        # Devolver el mismo session_id que recibimos (o el nuevo si se creó)
        return {
            "content": response,
            "session_id": session_id
        }
    except Exception as e:
        print(f"[Backend] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
