import time
import streamlit as st
from google import genai
from google.genai import types
import anthropic

st.set_page_config(page_title="LexAlert - Inteligencia Regulatoria", page_icon="⚖️", layout="wide")

st.title("⚖️ LexAlert: Sistema Inteligente de Monitoreo y Alerta Normativa")
st.subheader("Detección automatizada de obligaciones legales según el perfil empresarial (Fusión Gemini + Claude)")
st.markdown("---")

st.sidebar.header("🔑 Configuración de APIs")
gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password", placeholder="AQ.Ab8... o AIzaSy...")
claude_key = st.sidebar.text_input("Anthropic Claude API Key (opcional)", type="password", placeholder="sk-ant...")
forzar_demo_local = st.sidebar.checkbox("Modo presentación sin conexión / respaldo inmediato", value=False)

if not gemini_key and not forzar_demo_local:
    st.sidebar.warning("⚠️ Ingresa tu Google Gemini API Key o activa el modo presentación si hay corte SSL.")

st.header("🏢 1. Perfil de la Empresa")
st.caption("Ingresa los datos de la organización para que el grafo relacione el marco regulatorio aplicable en Colombia.")

col_p1, col_p2 = st.columns(2)

with col_p1:
    nombre_empresa = st.text_input("Nombre o Razón Social", "TechLogística S.A.S.")
    sector = st.selectbox(
        "Sector Económico",
        [
            "Comercio Electrónico y Tecnología",
            "Transporte y Logística",
            "Alimentos, Bebidas y Restaurantes",
            "Salud y Farmacéutico",
            "Manufactura e Industria",
            "Servicios Financieros / Fintech",
            "Construcción e Inmobiliario",
            "Otro"
        ]
    )
    ubicacion = st.text_input("Ciudad / Jurisdicción", "Bogotá D.C., Colombia")

with col_p2:
    empleados = st.number_input("Número de empleados", min_value=1, max_value=10000, value=25)
    ingresos_aprox = st.selectbox(
        "Tamaño empresarial (Decreto 957 de 2019)",
        ["Microempresa", "Pequeña Empresa", "Mediana Empresa", "Gran Empresa"]
    )
    enfoque_consulta = st.multiselect(
        "Áreas regulatorias de interés prioritario",
        ["Laboral y Seguridad Social", "Protección de Datos (Habeas Data)", "Tributario y Facturación (DIAN)", "Protección al Consumidor", "Ambiental y Sanitario", "Societario / Cumplimiento (SAGRILAFT/PTEE)"],
        default=["Laboral y Seguridad Social", "Protección de Datos (Habeas Data)", "Tributario y Facturación (DIAN)"]
    )

actividad = st.text_area(
    "Descripción detallada de la actividad económica y operaciones",
    "Venta de productos tecnológicos a través de plataforma web propia, procesamiento de pagos en línea, almacenamiento de bases de datos de clientes y despacho de mercancía a nivel nacional con flota propia y terceros.",
    height=100
)

st.markdown("---")


def generar_mapeo_respaldo_local(nombre: str, sec: str, tam: str, emp: int, ubi: str, areas: list, op: str) -> str:
    areas_txt = ", ".join(areas) if areas else "Cumplimiento general"
    return f"""### Mapeo Normativo Inicial (Respaldo Local Activo) — {nombre}
- **Perfil evaluado:** {sec} | {tam} ({emp} empleados) | {ubi}
- **Áreas priorizadas:** {areas_txt}
- **Operación analizada:** {op}

1. **Ley 2101 de 2021 (Reducción gradual de la jornada laboral) y lineamientos laborales vigentes en Colombia (2024–2026)**
   - **Entidad emisora:** Congreso de la República / Ministerio del Trabajo.
   - **Nivel de Impacto:** Crítico.
   - **Justificación técnica:** Al contar con {emp} empleados en {ubi}, la empresa debe mantener ajustado su tope semanal de horas ordinarias, turnos operativos y reglamento interno sin desmejorar salarios ni prestaciones.

2. **Ley 1581 de 2012, Decreto 1074 de 2015 e instrucciones de la Superintendencia de Industria y Comercio (SIC) sobre Habeas Data y comercio electrónico**
   - **Entidad emisora:** Superintendencia de Industria y Comercio (SIC).
   - **Nivel de Impacto:** Crítico.
   - **Justificación técnica:** La captación y almacenamiento de bases de datos de clientes en plataforma web y pagos en línea exige autorización previa, política de tratamiento de datos, medidas de seguridad y actualización del RNBD cuando aplique.

3. **Estatuto Tributario y Resoluciones DIAN sobre Facturación Electrónica, Documento Soporte y Nómina Electrónica**
   - **Entidad emisora:** Dirección de Impuestos y Aduanas Nacionales (DIAN).
   - **Nivel de Impacto:** Alto.
   - **Justificación técnica:** La venta de bienes/servicios en línea y el pago de {emp} trabajadores requieren emisión válida de factura electrónica, documento soporte a no obligados/proveedores y transmisión oportuna de nómina electrónica.

4. **Estatuto del Consumidor (Ley 1480 de 2011) y reglas de comercio electrónico y transporte de carga en Colombia**
   - **Entidad emisora:** Superintendencia de Industria y Comercio (SIC) / Ministerio de Transporte.
   - **Nivel de Impacto:** Alto.
   - **Justificación técnica:** La venta digital y el despacho de mercancías con flota propia y terceros activan deberes de información clara, reversión del pago/derecho de retracto cuando aplique, garantía legal y condiciones de transporte."""


def generar_auditoria_respaldo_local(nombre: str, sec: str, emp: int, ubi: str, mapeo: str) -> str:
    return f"""### Tablero Ejecutivo de Alertas Auditadas — {nombre} ({ubi})

| Prioridad | Marco Normativo | Riesgo Principal | Acción Inmediata |
| :--- | :--- | :--- | :--- |
| **1. Inmediata** | Jornada laboral y nómina ({emp} trabajadores) | Sobrecostos por horas extras no previstas y sanciones laborales | Auditar turnos, contratos y liquidación de nómina electrónica |
| **2. Inmediata** | Protección de Datos / SIC (clientes web y pagos) | Multas administrativas SIC y requerimientos por fugas o falta de autorización | Revisar avisos de privacidad, consentimientos en checkout y seguridad de bases |
| **3. Alta** | Facturación Electrónica y Soporte DIAN | Rechazo de costos/deducciones y sanciones tributarias | Verificar validación previa DIAN y documentos soporte de logística/terceros |
| **4. Alta** | Comercio Electrónico y Transporte | Reclamaciones de consumidores, devoluciones y contingencias en ruta | Auditar términos de venta online, garantías y contratos con transportadores |

### Detalle de Obligaciones y Controles para el Área Legal ({sec})
- **Hacer:** Documentar políticas de tratamiento de datos, validar flujos de facturación electrónica y formalizar contratos/responsabilidades con proveedores logísticos terceros.
- **No hacer:** Recolectar datos personales sin finalidad clara ni operar esquemas horarios que superen el límite legal vigente sin compensación debida.

---
*Nota de continuidad:* {mapeo.splitlines()[0]}"""


def llamar_gemini_con_reintentos(api_key: str, prompt: str, intentos: int = 3) -> str:
    client_gemini = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=25,
            transport="rest",
        ),
    )
    ultimo_error = None
    for intento in range(1, intentos + 1):
        try:
            res_gemini = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if res_gemini and res_gemini.text:
                return res_gemini.text
            raise RuntimeError("Gemini devolvió una respuesta vacía.")
        except Exception as e:
            ultimo_error = e
            texto_error = str(e).lower()
            reintentable = (
                "503" in texto_error
                or "unavailable" in texto_error
                or "high demand" in texto_error
                or "timeout" in texto_error
                or "timed out" in texto_error
                or "handshake" in texto_error
                or "ssl" in texto_error
                or "connection" in texto_error
            )
            if reintentable and intento < intentos:
                time.sleep(2)
                continue
            raise ultimo_error


if st.button("🔍 Analizar Perfil y Generar Alertas Normativas (Gemini ➔ Claude)", type="primary"):
    if not gemini_key and not forzar_demo_local:
        st.error("❌ Falta la Google Gemini API Key en la barra lateral izquierda.")
    else:
        usó_respaldo_red = forzar_demo_local
        error_red = ""

        prompt_gemini = f"""
        Actúa como el motor de descubrimiento y pertinencia normativa de LexAlert (Gemini) especializado en legislación colombiana (actualizado a 2026).

        Analiza el siguiente perfil empresarial:
        - Empresa: {nombre_empresa}
        - Sector: {sector}
        - Tamaño: {ingresos_aprox} ({empleados} empleados)
        - Ubicación: {ubicacion}
        - Áreas prioritarias: {', '.join(enfoque_consulta)}
        - Operación detallada: {actividad}

        TAREA:
        Identifica y filtra las 4 normativas reales más críticas y recientes (leyes, decretos, resoluciones o circulares en Colombia vigentes o con hitos clave entre 2024 y 2026, como la reforma laboral/reducción de jornada Ley 2101, normativa SIC de datos, facturación electrónica DIAN, comercio electrónico, etc.) que afectan DIRECTAMENTE a esta empresa por su operación y tamaño.

        Para cada una de las 4 normas entrega:
        1. Nombre oficial y entidad emisora.
        2. Nivel de Impacto (Crítico / Alto / Medio).
        3. Justificación técnica de por qué el perfil de esta empresa activa esta norma.
        """

        with st.spinner("Paso 1/2: Cruzando el perfil empresarial con el marco normativo colombiano..."):
            if not forzar_demo_local:
                try:
                    analisis_gemini = llamar_gemini_con_reintentos(gemini_key, prompt_gemini)
                except Exception as e:
                    usó_respaldo_red = True
                    error_red = str(e)
                    analisis_gemini = generar_mapeo_respaldo_local(
                        nombre_empresa, sector, ingresos_aprox, empleados, ubicacion, enfoque_consulta, actividad
                    )
            else:
                analisis_gemini = generar_mapeo_respaldo_local(
                    nombre_empresa, sector, ingresos_aprox, empleados, ubicacion, enfoque_consulta, actividad
                )

        with st.spinner("Paso 2/2: Auditando el análisis y estructurando obligaciones para el abogado..."):
            prompt_auditoria = f"""Actúa como el Auditor Jurídico Senior de LexAlert.
Acabas de recibir el siguiente mapeo normativo inicial generado para la empresa '{nombre_empresa}' ({sector}, {empleados} empleados, ubicada en {ubicacion}).

Mapeo preliminar:
{analisis_gemini}

TAREA DE AUDITORÍA:
1. Valida que las normas citadas correspondan realmente al ordenamiento jurídico colombiano y corrige o precisa cualquier imprecisión técnica para evitar alucinaciones.
2. Construye un **Tablero Ejecutivo de Alertas** ordenado por urgencia.
3. Para cada norma, extrae las **obligaciones concretas de hacer/no hacer**, el riesgo o sanción por incumplimiento, y la **acción recomendada inmediata** para el equipo legal de la empresa."""

            analisis_auditoria = ""
            auditoria_completada = False

            if not usó_respaldo_red and claude_key.strip():
                try:
                    client_claude = anthropic.Anthropic(api_key=claude_key.strip(), timeout=25.0)
                    message = client_claude.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": prompt_auditoria}],
                    )
                    analisis_auditoria = message.content[0].text
                    auditoria_completada = True
                except Exception:
                    auditoria_completada = False

            if not usó_respaldo_red and not auditoria_completada:
                try:
                    analisis_auditoria = llamar_gemini_con_reintentos(gemini_key, prompt_auditoria)
                    auditoria_completada = True
                except Exception as e2:
                    usó_respaldo_red = True
                    error_red = str(e2)

            if not auditoria_completada:
                analisis_auditoria = generar_auditoria_respaldo_local(
                    nombre_empresa, sector, empleados, ubicacion, analisis_gemini
                )

        st.header(f"🔔 Reporte de Cumplimiento Normativo: {nombre_empresa}")

        tab1, tab2 = st.tabs(["🛡️ Alerta Final Auditada", "🔗 Mapeo y Pertinencia Inicial"])

        with tab1:
            if usó_respaldo_red and not forzar_demo_local:
                st.warning(f"⚠️ Se activó el respaldo local por bloqueo/timeout SSL en la red ({error_red}). El reporte sigue disponible para tu presentación.")
            elif forzar_demo_local:
                st.info("ℹ️ Modo presentación / respaldo local activado.")
            else:
                st.success("✅ Auditoría completada.")
            st.markdown(analisis_auditoria)

        with tab2:
            st.info("ℹ️ Cruce normativo inicial según el perfil ingresado.")
            st.markdown(analisis_gemini)
