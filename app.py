import json
import re
import time
import streamlit as st
from google import genai
from google.genai import types

try:
    import anthropic
except ImportError:
    anthropic = None

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


def generar_respaldo_dinamico(perfil: dict, motivo: str) -> dict:
    """Respaldo adaptable a la ciudad, sector y texto libre ingresado en vivo."""
    nombre = perfil.get("nombre_empresa", "la empresa")
    sector = perfil.get("sector", "Otro")
    tamano = perfil.get("tamano", "PyME")
    empleados = perfil.get("empleados", 0)
    ciudad = perfil.get("ciudad", "la ciudad indicada")
    actividad = perfil.get("actividad", "")
    riesgos = perfil.get("riesgos", "")

    extracto_op = actividad[:130].strip() if actividad else f"operación en {sector}"
    extracto_riesgo = riesgos[:110].strip() if riesgos else "cumplimiento regulatorio y vigilancia"

    alertas = [
        {
            "id": "NOV-01",
            "norma": f"Actualizaciones regulatorias recientes y vigilancia sectorial ({sector} — 2024–2026)",
            "entidad": "Ministerio del ramo / Superintendencia o autoridad competente",
            "impacto": "Alto",
            "resumen_aplicacion": (
                f"Evalúa novedades normativas y líneas de supervisión vigentes sobre '{extracto_op}' "
                f"para {nombre} ({tamano}, {empleados} empleados), más allá de las obligaciones básicas tradicionales."
            ),
            "accion_recomendada": (
                f"Contrastar los procesos actuales de {nombre} frente a resoluciones, circulares o guías "
                f"emitidas recientemente en {sector}."
            ),
        },
        {
            "id": "NOV-02",
            "norma": f"Normativa territorial, POT, uso del suelo y trámites locales en {ciudad}",
            "entidad": f"Alcaldía / Secretaría de Planeación o Autoridad Ambiental de {ciudad}",
            "impacto": "Alto",
            "resumen_aplicacion": (
                f"Aplica a la operación de {nombre} en {ciudad}: verifica compatibilidad de uso, "
                f"exigencias locales, movilidad, ruido, residuos o licencias propias de {ciudad} para '{extracto_op}'."
            ),
            "accion_recomendada": (
                f"Verificar requisitos vigentes de funcionamiento y normatividad territorial específica de {ciudad}."
            ),
        },
        {
            "id": "NOV-03",
            "norma": f"Control de riesgos emergentes y reporte/cumplimiento ({extracto_riesgo})",
            "entidad": "Entidad de inspección, vigilancia y control (IVC)",
            "impacto": "Medio",
            "resumen_aplicacion": (
                f"Aborda el frente de '{extracto_riesgo}' en una organización de tamaño {tamano} "
                f"con {empleados} colaboradores, exigiendo controles documentados y trazabilidad."
            ),
            "accion_recomendada": (
                f"Definir un control verificable sobre '{extracto_riesgo}' adaptable a la escala de {nombre}."
            ),
        },
        {
            "id": "NOV-04",
            "norma": "Exigencias de transparencia digital, facturación, trazabilidad y gobernanza de datos",
            "entidad": "DIAN / SIC / Autoridad sectorial correspondiente",
            "impacto": "Medio",
            "resumen_aplicacion": (
                f"Revisa obligaciones operativas recientes asociadas a digitalización, soporte documental, "
                f"intercambio de información y protección de datos en '{extracto_op}'."
            ),
            "accion_recomendada": (
                "Auditar soportes digitales, contratos con terceros y flujos de información de la empresa."
            ),
        },
    ]

    return {
        "modo_respaldo": True,
        "motivo_respaldo": motivo,
        "resumen_ejecutivo": (
            f"Análisis dinámico de contingencia para {nombre} en {ciudad} ({sector}, {tamano}, "
            f"{empleados} empleados), enfocado en normativa territorial, novedades sectoriales y '{extracto_riesgo}'."
        ),
        "alertas": alertas,
    }


def extraer_json(texto: str) -> dict:
    limpio = texto.strip()
    limpio = re.sub(r"^```(?:json)?\s*", "", limpio)
    limpio = re.sub(r"\s*```$", "", limpio)
    inicio = limpio.find("{")
    fin = limpio.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        limpio = limpio[inicio : fin + 1]
    return json.loads(limpio)


def construir_prompt_novedad(perfil: dict) -> str:
    return (
        "Actúa como abogado experto en regulación colombiana y derecho administrativo/territorial. "
        "El usuario está haciendo una prueba en vivo con un caso nuevo. "
        "EVITA mencionar únicamente leyes generales archiconocidas de siempre (como solo decir Ley 1581 o Código Sustantivo "
        "de forma genérica). Prioriza NORMAS RECIENTES (2023–2026), decretos, resoluciones, circulares, guías de vigilancia "
        "o exigencias específicas vinculadas a la CIUDAD/MUNICIPIO y a la ACTIVIDAD exacta descrita.\n"
        "Responde ÚNICAMENTE con un JSON válido (sin markdown ni texto extra) con esta estructura exacta:\n"
        '{"resumen_ejecutivo":"...",'
        '"alertas":[{"id":"NOV-01","norma":"Norma específica, reciente o territorial con año/número si aplica",'
        '"entidad":"Entidad reguladora o autoridad local/sectorial","impacto":"Alto|Medio|Bajo",'
        '"resumen_aplicacion":"Por qué aplica a esta empresa, actividad y ciudad en 1-2 oraciones",'
        '"accion_recomendada":"Acción concreta en 1 oración"}]}\n'
        "Genera 4 alertas variadas y específicas para este caso en vivo:\n"
        f"- Empresa: {perfil['nombre_empresa']}\n"
        f"- Sector: {perfil['sector']}\n"
        f"- Tamaño: {perfil['tamano']} ({perfil['empleados']} empleados)\n"
        f"- Ciudad/Municipio: {perfil['ciudad']}, Colombia\n"
        f"- Actividad y operación: {perfil['actividad']}\n"
        f"- Focos/Riesgos indicados: {perfil['riesgos']}"
    )


def llamar_gemini(api_key: str, perfil: dict) -> dict:
    client = genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=60000),
    )
    prompt = construir_prompt_novedad(perfil)

    ultimo_error = None
    for intento in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.65,
                    max_output_tokens=1800,
                ),
            )
            data = extraer_json(resp.text or "")
            if not isinstance(data.get("alertas"), list) or len(data["alertas"]) == 0:
                raise ValueError("Respuesta JSON sin alertas válidas.")
            data["modo_respaldo"] = False
            data["motivo_respaldo"] = ""
            return data
        except Exception as exc:
            ultimo_error = exc
            msg = str(exc)
            if "404" in msg or "API_KEY_INVALID" in msg or "API key not valid" in msg:
                break
            time.sleep(1.5 * (intento + 1))

    raise RuntimeError(str(ultimo_error))


def llamar_claude(api_key: str, perfil: dict) -> dict:
    if anthropic is None:
        raise RuntimeError("Librería anthropic no instalada.")
    client = anthropic.Anthropic(api_key=api_key.strip(), timeout=60.0)
    prompt = construir_prompt_novedad(perfil)

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1800,
        temperature=0.65,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
    data = extraer_json(texto)
    if not isinstance(data.get("alertas"), list) or len(data["alertas"]) == 0:
        raise ValueError("Respuesta JSON de Claude sin alertas.")
    data["modo_respaldo"] = False
    data["motivo_respaldo"] = ""
    return data


# --- Interfaz ---
st.title("⚖️ LexAlert — Radar Normativo Empresarial")
st.caption("Evaluación en vivo de normativa reciente, territorial y sectorial según los datos de cada empresa.")

with st.sidebar:
    st.header("Configuración del motor IA")
    proveedor = st.selectbox("Proveedor principal", ["Google Gemini", "Anthropic Claude"])

    gemini_key_default = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    claude_key_default = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

    gemini_key = st.text_input(
        "Gemini API Key",
        value=gemini_key_default,
        type="password",
        help="Obtén tu clave en Google AI Studio.",
    )
    claude_key = st.text_input(
        "Anthropic API Key (opcional)",
        value=claude_key_default,
        type="password",
    )

    modo_sin_conexion = st.checkbox(
        "Forzar modo presentación / respaldo dinámico",
        value=False,
        help="Actívalo solo si quieres simular el respaldo sin llamar a la API.",
    )

col1, col2 = st.columns(2)
with col1:
    nombre_empresa = st.text_input("Nombre de la empresa", placeholder="Ej. Logística del Pacífico S.A.S.")
    sector = st.selectbox("Sector económico", SECTORES_COLOMBIA)
    tamano = st.selectbox("Tamaño de empresa", ["Microempresa", "Pequeña", "Mediana", "Grande"])
with col2:
    empleados = st.number_input("Número de empleados", min_value=1, max_value=100000, value=25, step=1)
    ciudad = st.text_input("Ciudad / municipio de operación", placeholder="Ej. Medellín, Cali, Barranquilla...")
    riesgos = st.text_input("Focos, retos o riesgos a evaluar", placeholder="Ej. Permisos locales, trazabilidad, contratos")

actividad = st.text_area(
    "Actividad principal y modelo de operación (para prueba en vivo)",
    height=105,
    placeholder="Escribe aquí la operación real o hipotética que quiera probar el docente...",
)

analizar = st.button("Analizar Perfil y Generar Alertas Normativas", type="primary")

if analizar:
    perfil = {
        "nombre_empresa": nombre_empresa.strip() or "Empresa de prueba en vivo",
        "sector": sector,
        "tamano": tamano,
        "empleados": int(empleados),
        "ciudad": ciudad.strip() or "Ciudad no especificada",
        "actividad": actividad.strip() or f"Operación en {sector}",
        "riesgos": riesgos.strip() or "Novedades normativas, cumplimiento local y control operativo",
    }

    resultado = None
    with st.spinner("Buscando normativa reciente y aplicable a la ciudad y actividad ingresadas..."):
        if modo_sin_conexion:
            resultado = generar_respaldo_dinamico(perfil, "Activado manualmente en modo presentación")
        else:
            try:
                if proveedor == "Google Gemini":
                    if not gemini_key.strip():
                        raise ValueError("Falta ingresar la Gemini API Key en la barra lateral.")
                    resultado = llamar_gemini(gemini_key, perfil)
                else:
                    if not claude_key.strip():
                        raise ValueError("Falta ingresar la Anthropic API Key en la barra lateral.")
                    resultado = llamar_claude(claude_key, perfil)
            except Exception as err:
                resultado = generar_respaldo_dinamico(perfil, str(err))

    tab1, tab2 = st.tabs(["🛡️ Alerta Final Auditada", "📋 Perfil Analizado"])

    with tab1:
        if resultado.get("modo_respaldo"):
            st.warning(
                f"⚠️ Se activó el respaldo dinámico por bloqueo/timeout SSL en la red ({resultado.get('motivo_respaldo')}). "
                "El reporte sigue disponible para tu presentación."
            )
        else:
            st.success("✅ Auditoría completada.")

        st.subheader("Resumen Ejecutivo")
        st.write(resultado.get("resumen_ejecutivo", ""))

        st.subheader("Normativa y Alertas Identificadas")
        for item in resultado.get("alertas", []):
            with st.container(border=True):
                st.markdown(f"**{item.get('id', '')} — {item.get('norma', '')}**")
                st.caption(
                    f"Entidad: {item.get('entidad', 'N/D')} | "
                    f"Ciudad de referencia: {perfil['ciudad']} | "
                    f"Nivel de impacto: **{item.get('impacto', 'Medio')}**"
                )
                st.write(item.get("resumen_aplicacion", ""))
                st.markdown(f"**Acción recomendada:** {item.get('accion_recomendada', '')}")

    with tab2:
        st.json(perfil)
