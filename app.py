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


def generar_respaldo_empresa(perfil: dict, motivo: str) -> dict:
    nombre = perfil.get("nombre_empresa", "la empresa")
    sector = perfil.get("sector", "Otro")
    tamano = perfil.get("tamano", "PyME")
    empleados = perfil.get("empleados", 0)
    ciudad = perfil.get("ciudad", "Colombia")
    actividad = perfil.get("actividad", "")
    riesgos = perfil.get("riesgos", "")

    alertas = [
        {
            "id": "LEX-01",
            "norma": f"Actualizaciones normativas y regulación sectorial vigente ({sector})",
            "entidad": "Ministerio sectorial / Superintendencia competente",
            "impacto": "Alto",
            "obligaciones": (
                f"Alinear los procesos de {nombre} en {ciudad} con las exigencias vigentes del sector {sector} "
                f"y documentar los controles operativos exigidos para una organización de tamaño {tamano}."
            ),
            "que_hacer": (
                f"Realizar un diagnóstico de brechas normativas en {nombre} sobre '{actividad[:120]}' "
                "y asignar responsables por cada requisito aplicable."
            ),
        },
        {
            "id": "LEX-02",
            "norma": "Normativa laboral, jornada y gestión de riesgos en el entorno de trabajo",
            "entidad": "Ministerio del Trabajo",
            "impacto": "Alto" if empleados >= 10 else "Medio",
            "obligaciones": (
                f"Cumplir con las reglas vigentes de jornada laboral, contratación y seguridad y salud "
                f"para el equipo de {empleados} colaboradores de {nombre}."
            ),
            "que_hacer": (
                "Revisar turnos, políticas internas, afiliaciones y planes de prevención de riesgos laborales."
            ),
        },
        {
            "id": "LEX-03",
            "norma": "Protección de datos personales, privacidad y gobernanza de información",
            "entidad": "Superintendencia de Industria y Comercio (SIC)",
            "impacto": "Alto",
            "obligaciones": (
                "Contar con políticas de tratamiento de datos actualizadas, autorizaciones de titulares "
                "y medidas de seguridad acordes con el manejo de información de clientes, proveedores y empleados."
            ),
            "que_hacer": (
                "Actualizar avisos de privacidad, inventarios de bases de datos y acuerdos de confidencialidad."
            ),
        },
        {
            "id": "LEX-04",
            "norma": "Cumplimiento tributario, facturación electrónica y gestión de riesgos prioritarios",
            "entidad": "DIAN / Entidad de vigilancia correspondiente",
            "impacto": "Medio",
            "obligaciones": (
                f"Mantener al día soportes contables, facturación y controles frente a los riesgos indicados "
                f"({riesgos[:110] if riesgos else 'cumplimiento general'})."
            ),
            "que_hacer": (
                "Establecer un calendario mensual de control y auditoría interna sobre las obligaciones críticas."
            ),
        },
    ]

    return {
        "modo_respaldo": True,
        "motivo_respaldo": motivo,
        "resumen_ejecutivo": (
            f"Análisis normativo para {nombre} ({sector}, {tamano}, {empleados} empleados en {ciudad}). "
            f"Enfoque en normativa nueva y vigente, obligaciones clave y plan de acción para la empresa."
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


def construir_prompt_empresa(perfil: dict) -> str:
    return (
        "Eres un abogado experto en regulación empresarial en Colombia. "
        "Analiza la empresa y entrega NORMAS NUEVAS y vigentes aplicables a su actividad y sector, "
        "detallando con claridad las obligaciones de la empresa y qué debe hacer concretamente. "
        "Responde ÚNICAMENTE con un JSON válido (sin markdown ni texto adicional) con esta estructura:\n"
        '{"resumen_ejecutivo":"...",'
        '"alertas":[{"id":"LEX-01","norma":"Nombre, número y año de la norma nueva o vigente",'
        '"entidad":"Entidad reguladora","impacto":"Alto|Medio|Bajo",'
        '"obligaciones":"Obligaciones concretas que impone a la empresa",'
        '"que_hacer":"Qué debe hacer paso a paso la empresa para cumplir"}]}\n'
        "Genera 4 normas relevantes y actuales para:\n"
        f"- Empresa: {perfil['nombre_empresa']}\n"
        f"- Sector: {perfil['sector']}\n"
        f"- Tamaño: {perfil['tamano']} ({perfil['empleados']} empleados)\n"
        f"- Ciudad: {perfil['ciudad']}\n"
        f"- Actividad: {perfil['actividad']}\n"
        f"- Riesgos / prioridades: {perfil['riesgos']}"
    )


def llamar_gemini(api_key: str, perfil: dict) -> dict:
    client = genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=60000),
    )
    prompt = construir_prompt_empresa(perfil)

    ultimo_error = None
    for intento in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=2000,
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
    prompt = construir_prompt_empresa(perfil)

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.4,
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
st.caption("Normas nuevas, obligaciones y plan de acción adaptado a tu empresa.")

with st.sidebar:
    st.header("Configuración IA")
    proveedor = st.selectbox("Proveedor", ["Google Gemini", "Anthropic Claude"])

    gemini_key_default = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    claude_key_default = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

    gemini_key = st.text_input(
        "Gemini API Key",
        value=gemini_key_default,
        type="password",
    )
    claude_key = st.text_input(
        "Anthropic API Key",
        value=claude_key_default,
        type="password",
    )

    modo_respaldo_manual = st.checkbox(
        "Modo respaldo local",
        value=False,
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

    with st.spinner("Analizando normativa aplicable a la empresa..."):
        if modo_respaldo_manual:
            resultado = generar_respaldo_empresa(perfil, "Modo respaldo local activado")
        else:
            try:
                if proveedor == "Google Gemini":
                    if not gemini_key.strip():
                        raise ValueError("Ingresa tu Gemini API Key en la barra lateral.")
                    resultado = llamar_gemini(gemini_key, perfil)
                else:
                    if not claude_key.strip():
                        raise ValueError("Ingresa tu Anthropic API Key en la barra lateral.")
                    resultado = llamar_claude(claude_key, perfil)
            except Exception as err:
                resultado = generar_respaldo_empresa(perfil, str(err))

    tab1, tab2 = st.tabs(["🛡️ Alerta Final Auditada", "📋 Perfil Analizado"])

    with tab1:
        if resultado.get("modo_respaldo"):
            st.warning(
                f"⚠️ Se activó el respaldo local ({resultado.get('motivo_respaldo')})."
            )
        else:
            st.success("✅ Auditoría completada.")

        st.subheader("Resumen Ejecutivo")
        st.write(resultado.get("resumen_ejecutivo", ""))

        st.subheader("Normativa, Obligaciones y Qué Hacer")
        for item in resultado.get("alertas", []):
            with st.container(border=True):
                st.markdown(f"### {item.get('id', '')} — {item.get('norma', '')}")
                st.caption(
                    f"**Entidad:** {item.get('entidad', 'N/D')} | "
                    f"**Impacto:** {item.get('impacto', 'Medio')}"
                )
                st.markdown("**Obligaciones para la empresa:**")
                st.write(item.get("obligaciones", item.get("resumen_aplicacion", "")))
                st.markdown("**Qué debe hacer la empresa:**")
                st.write(item.get("que_hacer", item.get("accion_recomendada", "")))

    with tab2:
        st.subheader("Información de la Empresa")
        st.markdown(f"**Nombre:** {perfil['nombre_empresa']}")
        st.markdown(f"**Sector:** {perfil['sector']}")
        st.markdown(f"**Tamaño:** {perfil['tamano']}")
        st.markdown(f"**Número de empleados:** {perfil['empleados']}")
        st.markdown(f"**Ciudad:** {perfil['ciudad']}")
        st.markdown(f"**Actividad:** {perfil['actividad']}")
        st.markdown(f"**Riesgos o temas clave:** {perfil['riesgos']}")
