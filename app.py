import json
import re
import time
import streamlit as st
from google import genai
from google.genai import types

try:
    from groq import Groq
except ImportError:
    Groq = None

st.set_page_config(
    page_title="LexAlert | Radar Normativo Empresarial",
    page_icon="⚖️",
    layout="wide",
)

SECTORES_COLOMBIA = [
    "Tecnología y Software",
    "Salud y Farma",
    "Construcción e Infraestructura",
    "Comercio y Retail",
    "Alimentos y Bebidas",
    "Servicios Financieros / Fintech",
    "Transporte y Logística",
    "Manufactura e Industria",
    "Consultoría y Servicios Corporativos",
    "Otro",
]

MODELOS_NO_CHAT = (
    "whisper",
    "orpheus",
    "tts",
    "audio",
    "guard",
    "safeguard",
    "embed",
)


def extraer_json(texto: str) -> dict:
    limpio = texto.strip()
    limpio = re.sub(r"^```(?:json)?\s*", "", limpio)
    limpio = re.sub(r"\s*```$", "", limpio)
    inicio = limpio.find("{")
    fin = limpio.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        limpio = limpio[inicio : fin + 1]
    return json.loads(limpio)


def construir_prompt(perfil: dict) -> str:
    return (
        "Eres un abogado experto en regulación empresarial en Colombia. "
        "Analiza la empresa y entrega normativa real y vigente aplicable a su actividad, sector y ciudad, "
        "detallando obligaciones de la empresa y qué debe hacer concretamente. "
        "No inventes números de leyes, decretos ni resoluciones; si un número exacto no es seguro, "
        "indica el marco normativo y la entidad competente para verificación directa.\n"
        "Responde ÚNICAMENTE con un JSON válido (sin markdown ni texto adicional) con esta estructura:\n"
        '{"resumen_ejecutivo":"...",'
        '"alertas":[{"norma":"Norma o marco regulatorio aplicable",'
        '"entidad":"Entidad reguladora o de vigilancia",'
        '"impacto":"Alto|Medio|Bajo",'
        '"obligaciones":"Obligaciones concretas para la empresa",'
        '"que_hacer":"Acciones concretas que debe hacer la empresa"}]}\n'
        "Genera 4 puntos normativos para:\n"
        f"- Empresa: {perfil['nombre_empresa']}\n"
        f"- Sector: {perfil['sector']}\n"
        f"- Tamaño: {perfil['tamano']} ({perfil['empleados']} empleados)\n"
        f"- Ciudad: {perfil['ciudad']}\n"
        f"- Actividad: {perfil['actividad']}\n"
        f"- Riesgos / temas clave: {perfil['riesgos']}"
    )


def listar_modelos_chat_groq(client: Groq) -> list[str]:
    modelos = client.models.list()
    todos = [getattr(m, "id", "") for m in getattr(modelos, "data", []) if getattr(m, "id", "")]
    candidatos_chat = [
        mid
        for mid in todos
        if not any(bloqueado in mid.lower() for bloqueado in MODELOS_NO_CHAT)
    ]

    preferencias = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-4-scout-17b-16e-instruct",
        "llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it",
    ]

    ordenados = []
    for pref in preferencias:
        if pref in candidatos_chat and pref not in ordenados:
            ordenados.append(pref)
    for mid in candidatos_chat:
        if mid not in ordenados:
            ordenados.append(mid)

    if not ordenados:
        raise RuntimeError(f"No hay modelos de texto disponibles en Groq. Modelos vistos: {todos}")
    return ordenados[:5]


def consultar_gemini(api_key: str, perfil: dict, usar_busqueda: bool = False) -> dict:
    client = genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=60),
    )
    prompt = construir_prompt(perfil)
    tools = [{"google_search": {}}] if usar_busqueda else None

    ultimo_error = None
    for intento in range(2):
        try:
            resp = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.25,
                    max_output_tokens=1600,
                    tools=tools,
                ),
            )
            data = extraer_json(resp.text or "")
            if not isinstance(data.get("alertas"), list) or len(data["alertas"]) == 0:
                raise ValueError("Gemini no devolvió alertas estructuradas.")
            data["motor_usado"] = "Google Gemini (gemini-3.8-flash)"
            return data
        except Exception as exc:
            ultimo_error = exc
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg or "API_KEY_INVALID" in msg:
                break
            time.sleep(1.0 * (intento + 1))

    raise RuntimeError(str(ultimo_error))


def consultar_groq(api_key: str, perfil: dict) -> dict:
    if Groq is None:
        raise RuntimeError("La librería 'groq' no está instalada. Agrégala a requirements.txt.")
    client = Groq(api_key=api_key.strip(), timeout=45.0)
    candidatos = listar_modelos_chat_groq(client)
    prompt = construir_prompt(perfil)

    ultimo_error = None
    for modelo_activo in candidatos:
        try:
            resp = client.chat.completions.create(
                model=modelo_activo,
                temperature=0.25,
                max_tokens=1600,
                messages=[
                    {
                        "role": "system",
                        "content": "Respondes únicamente en JSON válido sin bloques markdown ni texto extra.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            texto = resp.choices[0].message.content or ""
            data = extraer_json(texto)
            if not isinstance(data.get("alertas"), list) or len(data["alertas"]) == 0:
                raise ValueError(f"{modelo_activo} no devolvió alertas estructuradas.")
            data["motor_usado"] = f"Groq ({modelo_activo})"
            return data
        except Exception as exc:
            ultimo_error = exc
            continue

    raise RuntimeError(f"Ningún modelo de chat en Groq respondió. Último error: {ultimo_error}")


def auditar_con_segunda_ia(proveedor_secundario: str, api_key: str, perfil: dict, borrador: dict) -> str:
    resumen_borrador = json.dumps(borrador, ensure_ascii=False)
    prompt_auditoria = (
        "Revisa este borrador de análisis normativo para una empresa en Colombia y entrega una nota breve "
        "de auditoría (máximo 4 líneas) indicando si las obligaciones y acciones son coherentes con el perfil "
        "y qué punto conviene validar primero.\n"
        f"Perfil: {json.dumps(perfil, ensure_ascii=False)}\n"
        f"Borrador: {resumen_borrador}"
    )
    if proveedor_secundario == "Groq" and Groq is not None and api_key.strip():
        client = Groq(api_key=api_key.strip(), timeout=35.0)
        for modelo_activo in listar_modelos_chat_groq(client):
            try:
                resp = client.chat.completions.create(
                    model=modelo_activo,
                    temperature=0.2,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt_auditoria}],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                continue
        return ""

    if proveedor_secundario == "Google Gemini" and api_key.strip():
        client = genai.Client(api_key=api_key.strip(), http_options=types.HttpOptions(timeout=40))
        resp = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt_auditoria,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=300),
        )
        return (resp.text or "").strip()

    return ""


# --- Interfaz ---
st.title("⚖️ LexAlert — Radar Normativo Empresarial")
st.caption("Motor dual con Google Gemini y Groq: respaldo automático entre IAs y auditoría cruzada.")

with st.sidebar:
    st.header("Configuración Dual (Gemini + Groq)")
    estrategia = st.selectbox(
        "Modo de trabajo",
        [
            "Cascada automática (Gemini primero, si falla usa Groq)",
            "Cascada inversa (Groq primero, si falla usa Gemini)",
            "Solo Google Gemini",
            "Solo Groq",
        ],
    )

    gemini_key_default = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    groq_key_default = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    gemini_key = st.text_input("Gemini API Key", value=gemini_key_default, type="password")
    groq_key = st.text_input("Groq API Key", value=groq_key_default, type="password")

    usar_busqueda_gemini = st.checkbox(
        "Activar búsqueda web en Gemini",
        value=False,
        help="Puede hacer más lenta la respuesta en Gemini; déjalo desactivado si ves errores 504 o de cuota.",
    )
    auditoria_cruzada = st.checkbox(
        "Revisión cruzada con la segunda IA",
        value=True,
        help="Si ambas claves están presentes, la segunda IA revisa brevemente el resultado.",
    )

col1, col2 = st.columns(2)
with col1:
    nombre_empresa = st.text_input("Nombre de la empresa", placeholder="Ej. Salud S.A.S.")
    sector = st.selectbox("Sector económico", SECTORES_COLOMBIA)
    tamano = st.selectbox("Tamaño de empresa", ["Microempresa", "Pequeña", "Mediana", "Grande"])
with col2:
    empleados = st.number_input("Número de empleados", min_value=1, max_value=100000, value=15, step=1)
    ciudad = st.text_input("Ciudad", value="Bogotá")
    riesgos = st.text_input("Riesgos o temas clave", placeholder="Ej. Contratos, salubridad, datos")

actividad = st.text_area(
    "Actividad de la empresa",
    height=100,
    placeholder="Describe a qué se dedica la empresa...",
)

if st.button("Generar Análisis Normativo", type="primary"):
    perfil = {
        "nombre_empresa": nombre_empresa.strip() or "Empresa",
        "sector": sector,
        "tamano": tamano,
        "empleados": int(empleados),
        "ciudad": ciudad.strip() or "Colombia",
        "actividad": actividad.strip() or f"Operación en {sector}",
        "riesgos": riesgos.strip() or "Cumplimiento normativo general",
    }

    resultado = None
    errores = []
    nota_auditoria = ""

    with st.spinner("Consultando normativa con Gemini / Groq..."):
        if estrategia == "Solo Google Gemini":
            if not gemini_key.strip():
                errores.append("Falta la Gemini API Key.")
            else:
                try:
                    resultado = consultar_gemini(gemini_key, perfil, usar_busqueda_gemini)
                except Exception as exc:
                    errores.append(f"Gemini: {exc}")

        elif estrategia == "Solo Groq":
            if not groq_key.strip():
                errores.append("Falta la Groq API Key.")
            else:
                try:
                    resultado = consultar_groq(groq_key, perfil)
                except Exception as exc:
                    errores.append(f"Groq: {exc}")

        elif estrategia == "Cascada automática (Gemini primero, si falla usa Groq)":
            if gemini_key.strip():
                try:
                    resultado = consultar_gemini(gemini_key, perfil, usar_busqueda_gemini)
                except Exception as exc:
                    errores.append(f"Gemini: {exc}")
            else:
                errores.append("Gemini: sin clave ingresada.")

            if resultado is None:
                if groq_key.strip():
                    try:
                        resultado = consultar_groq(groq_key, perfil)
                    except Exception as exc:
                        errores.append(f"Groq: {exc}")
                else:
                    errores.append("Groq: sin clave ingresada.")

            if resultado and auditoria_cruzada and groq_key.strip() and "Groq" not in resultado.get("motor_usado", ""):
                try:
                    nota_auditoria = auditar_con_segunda_ia("Groq", groq_key, perfil, resultado)
                except Exception:
                    nota_auditoria = ""

        elif estrategia == "Cascada inversa (Groq primero, si falla usa Gemini)":
            if groq_key.strip():
                try:
                    resultado = consultar_groq(groq_key, perfil)
                except Exception as exc:
                    errores.append(f"Groq: {exc}")
            else:
                errores.append("Groq: sin clave ingresada.")

            if resultado is None:
                if gemini_key.strip():
                    try:
                        resultado = consultar_gemini(gemini_key, perfil, usar_busqueda_gemini)
                    except Exception as exc:
                        errores.append(f"Gemini: {exc}")
                else:
                    errores.append("Gemini: sin clave ingresada.")

            if resultado and auditoria_cruzada and gemini_key.strip() and "Gemini" not in resultado.get("motor_usado", ""):
                try:
                    nota_auditoria = auditar_con_segunda_ia("Google Gemini", gemini_key, perfil, resultado)
                except Exception:
                    nota_auditoria = ""

    tab1, tab2 = st.tabs(["🛡️ Análisis Normativo", "📋 Perfil de la Empresa"])

    with tab1:
        if resultado is None:
            st.error("No se pudo obtener respuesta de ninguna IA:\n\n" + "\n\n".join(errores))
        else:
            if errores and "Cascada" in estrategia:
                st.info(
                    f"Respondido por **{resultado.get('motor_usado')}** tras contingencia en el primer motor "
                    f"({errores[0]})."
                )
            else:
                st.success(f"✅ Análisis completado con **{resultado.get('motor_usado')}**.")

            if nota_auditoria:
                st.markdown("**Revisión cruzada:**")
                st.caption(nota_auditoria)

            st.subheader("Resumen Ejecutivo")
            st.write(resultado.get("resumen_ejecutivo", ""))

            st.subheader("Normativa, Obligaciones y Qué Hacer")
            for idx, item in enumerate(resultado.get("alertas", []), start=1):
                with st.container(border=True):
                    st.markdown(f"### {idx}. {item.get('norma', '')}")
                    st.caption(
                        f"**Entidad:** {item.get('entidad', 'No especificada')} | "
                        f"**Impacto:** {item.get('impacto', 'Medio')}"
                    )
                    st.markdown("**Obligaciones para la empresa:**")
                    st.write(item.get("obligaciones", ""))
                    st.markdown("**Qué debe hacer la empresa:**")
                    st.write(item.get("que_hacer", ""))

    with tab2:
        st.subheader("Información de la Empresa")
        st.markdown(f"**Nombre:** {perfil['nombre_empresa']}")
        st.markdown(f"**Sector:** {perfil['sector']}")
        st.markdown(f"**Tamaño:** {perfil['tamano']}")
        st.markdown(f"**Número de empleados:** {perfil['empleados']}")
        st.markdown(f"**Ciudad:** {perfil['ciudad']}")
        st.markdown(f"**Actividad:** {perfil['actividad']}")
        st.markdown(f"**Riesgos o temas clave:** {perfil['riesgos']}")
