"""
Sistema de traducción multi-agente secuencial.

Pipeline EN→FR→ES (cada agente recibe la salida del anterior):

    Texto en inglés
        │
        ▼
    [Agente 1] Traductor EN → FR
        │   (texto original + traducción al francés)
        ▼
    [Agente 2] Traductor FR → ES   (recibe SOLO la traducción al francés)
        │   (español traducido desde el francés)
        ▼
    [Agente 3] Revisor de calidad lingüística
        │
        ▼
    Resultado final (traducciones revisadas + informe de calidad)

Uso:
    # Opción A: variable de entorno
    export DEEPSEEK_API_KEY="sk-..."          # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
    # Opción B: archivo .env junto al script con la línea  DEEPSEEK_API_KEY=sk-...

    python traductor_multiagente.py "The quick brown fox jumps over the lazy dog."

    # Sin API key (modo demostración con respuestas simuladas):
    python traductor_multiagente.py --mock "Hello world"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Estado compartido que fluye por el pipeline
# ---------------------------------------------------------------------------

@dataclass
class EstadoTraduccion:
    """Contexto que cada agente recibe, enriquece y pasa al siguiente."""
    texto_original: str
    traduccion_fr: str | None = None
    traduccion_es: str | None = None
    revision: dict[str, Any] | None = None
    historial: list[dict[str, Any]] = field(default_factory=list)

    def registrar(self, agente: str, entrada: str, salida: str, segundos: float) -> None:
        self.historial.append({
            "agente": agente,
            "entrada": entrada,
            "salida": salida,
            "duracion_s": round(segundos, 2),
        })


# ---------------------------------------------------------------------------
# Cliente LLM (real o simulado)
# ---------------------------------------------------------------------------

class ClienteLLM(Protocol):
    def completar(self, sistema: str, mensaje: str) -> str: ...


# ---------------------------------------------------------------------------
# Configuración de DeepSeek
# La llave se lee de la variable de entorno DEEPSEEK_API_KEY, o de un archivo
# `.env` junto a este script (ver `.env.example`). Nunca la escribas en el código.
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELO_POR_DEFECTO = "deepseek-flash"


def cargar_env(ruta: str | None = None) -> None:
    """Carga variables desde un archivo .env (sin dependencias externas)."""
    ruta = ruta or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(ruta):
        return
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


class ClienteDeepSeek:
    """Cliente para la API de DeepSeek (compatible con el SDK de OpenAI)."""

    def __init__(self, modelo: str | None = None, max_tokens: int = 4096, pensar: bool = False):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("Instala el SDK: pip install openai") from e
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "Falta la llave de DeepSeek. Defínela en la variable de entorno "
                "DEEPSEEK_API_KEY o en un archivo .env (DEEPSEEK_API_KEY=sk-...), o usa --mock."
            )
        self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.modelo = modelo or os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODELO_POR_DEFECTO)
        self.max_tokens = max_tokens
        # El modo "thinking" está activo por defecto en DeepSeek: es más lento y
        # puede gastar los tokens en razonar y dejar `content` vacío. Para traducir
        # no hace falta, así que se desactiva salvo que se pida con --pensar.
        self.pensar = pensar

    def completar(self, sistema: str, mensaje: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.modelo,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": sistema},
                    {"role": "user", "content": mensaje},
                ],
                stream=False,
                extra_body={"thinking": {"type": "enabled" if self.pensar else "disabled"}},
            )
        except Exception as e:  # errores de red, llave inválida, saldo, modelo...
            raise SystemExit(f"\n[ERROR] Falló la llamada a DeepSeek: {type(e).__name__}: {e}")
        choice = resp.choices[0]
        texto = (choice.message.content or "").strip()
        if not texto:
            raise SystemExit(
                f"\n[ERROR] DeepSeek devolvió una respuesta vacía "
                f"(finish_reason={choice.finish_reason}). Prueba sin --pensar o con otro --modelo."
            )
        return texto


class ClienteSimulado:
    """Cliente falso para probar el flujo sin red ni API key."""

    def completar(self, sistema: str, mensaje: str) -> str:
        if "de inglés a francés" in sistema:
            return f"[FR] {self._etiqueta(mensaje, 'texto_original')}"
        if "de francés a español" in sistema:
            return f"[ES←{self._etiqueta(mensaje, 'texto_frances')}]"
        # Revisor
        return json.dumps({
            "frances": {"puntuacion": 8, "problemas": ["(simulado) sin revisión real"],
                        "traduccion_corregida": "[FR revisado] ..."},
            "espanol": {"puntuacion": 8, "problemas": ["(simulado) sin revisión real"],
                        "traduccion_corregida": "[ES revisado] ..."},
            "consistencia": "(simulado) Ambas traducciones conservan el sentido.",
            "veredicto": "APROBADO",
        }, ensure_ascii=False)

    @staticmethod
    def _etiqueta(mensaje: str, tag: str) -> str:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", mensaje, re.S)
        return m.group(1) if m else mensaje


# ---------------------------------------------------------------------------
# Agentes
# ---------------------------------------------------------------------------

class Agente(ABC):
    nombre: str = "Agente"
    prompt_sistema: str = ""

    def __init__(self, llm: ClienteLLM):
        self.llm = llm

    def ejecutar(self, estado: EstadoTraduccion) -> EstadoTraduccion:
        entrada = self.construir_entrada(estado)
        t0 = time.perf_counter()
        salida = self.llm.completar(self.prompt_sistema, entrada)
        estado.registrar(self.nombre, entrada, salida, time.perf_counter() - t0)
        self.actualizar_estado(estado, salida)
        return estado

    @abstractmethod
    def construir_entrada(self, estado: EstadoTraduccion) -> str: ...

    @abstractmethod
    def actualizar_estado(self, estado: EstadoTraduccion, salida: str) -> None: ...


class TraductorFrances(Agente):
    nombre = "Traductor EN→FR"
    prompt_sistema = (
        "Eres un traductor profesional de inglés a francés. Traduce el texto con "
        "fidelidad al significado, tono y registro, usando francés natural e idiomático. "
        "Devuelve ÚNICAMENTE la traducción, sin comentarios ni comillas."
    )

    def construir_entrada(self, estado):
        return f"<texto_original>\n{estado.texto_original}\n</texto_original>"

    def actualizar_estado(self, estado, salida):
        estado.traduccion_fr = salida


class TraductorEspanol(Agente):
    nombre = "Traductor FR→ES"
    prompt_sistema = (
        "Eres un traductor profesional de francés a español. Traduce el texto en "
        "francés al español neutro, natural e idiomático, con fidelidad al significado, "
        "tono y registro. "
        "Devuelve ÚNICAMENTE la traducción al español, sin comentarios ni comillas."
    )

    def construir_entrada(self, estado):
        # Solo recibe la salida del agente anterior (francés), no el texto original.
        if estado.traduccion_fr is None:
            raise RuntimeError("El traductor FR→ES requiere la salida del traductor EN→FR.")
        return f"<texto_frances>\n{estado.traduccion_fr}\n</texto_frances>"

    def actualizar_estado(self, estado, salida):
        estado.traduccion_es = salida


class RevisorCalidad(Agente):
    nombre = "Revisor de calidad"
    prompt_sistema = (
        "Eres un revisor experto en calidad lingüística (inglés, francés y español). "
        "La traducción al español se hizo a partir del francés (cadena EN→FR→ES), así que "
        "detecta errores que se hayan arrastrado o añadido en esa cadena. "
        "Compara cada traducción con el original y evalúa precisión, fluidez, gramática, "
        "terminología y registro. Corrige lo necesario. Responde SOLO con un JSON válido "
        "con esta estructura exacta:\n"
        '{"frances": {"puntuacion": <1-10>, "problemas": [<str>], "traduccion_corregida": <str>},\n'
        ' "espanol": {"puntuacion": <1-10>, "problemas": [<str>], "traduccion_corregida": <str>},\n'
        ' "consistencia": <str>, "veredicto": "APROBADO" | "REQUIERE_REVISION"}'
    )

    def construir_entrada(self, estado):
        if estado.traduccion_fr is None or estado.traduccion_es is None:
            raise RuntimeError("El revisor requiere ambas traducciones.")
        return (
            f"<texto_original>\n{estado.texto_original}\n</texto_original>\n\n"
            f"<traduccion_frances>\n{estado.traduccion_fr}\n</traduccion_frances>\n\n"
            f"<traduccion_espanol>\n{estado.traduccion_es}\n</traduccion_espanol>"
        )

    def actualizar_estado(self, estado, salida):
        estado.revision = self._parsear_json(salida)

    @staticmethod
    def _parsear_json(texto: str) -> dict[str, Any]:
        m = re.search(r"\{.*\}", texto, re.S)  # tolera texto o ```json alrededor
        try:
            return json.loads(m.group(0) if m else texto)
        except json.JSONDecodeError:
            return {"error": "Respuesta del revisor no es JSON válido", "crudo": texto}


# ---------------------------------------------------------------------------
# Orquestador secuencial
# ---------------------------------------------------------------------------

class PipelineSecuencial:
    def __init__(self, agentes: list[Agente], verbose: bool = True):
        self.agentes = agentes
        self.verbose = verbose

    def ejecutar(self, texto: str) -> EstadoTraduccion:
        estado = EstadoTraduccion(texto_original=texto.strip())
        for i, agente in enumerate(self.agentes, 1):
            if self.verbose:
                print(f"▶ Paso {i}/{len(self.agentes)}: {agente.nombre}...", flush=True)
            estado = agente.ejecutar(estado)  # salida de uno = entrada del siguiente
            if self.verbose:
                print(f"  ✓ listo en {estado.historial[-1]['duracion_s']} s", flush=True)
        return estado


def crear_pipeline(llm: ClienteLLM, verbose: bool = True) -> PipelineSecuencial:
    return PipelineSecuencial(
        [TraductorFrances(llm), TraductorEspanol(llm), RevisorCalidad(llm)],
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def imprimir_resultado(estado: EstadoTraduccion) -> None:
    sep = "─" * 60
    print(f"\n{sep}\nORIGINAL (EN)\n{estado.texto_original}")
    print(f"\n{sep}\nFRANCÉS (borrador)\n{estado.traduccion_fr}")
    print(f"\n{sep}\nESPAÑOL (borrador)\n{estado.traduccion_es}")
    rev = estado.revision or {}
    print(f"\n{sep}\nREVISIÓN DE CALIDAD")
    if "error" in rev:
        print(rev["error"]); print(rev.get("crudo", ""))
        return
    for clave, etiqueta in (("frances", "Francés"), ("espanol", "Español")):
        r = rev.get(clave, {})
        print(f"\n• {etiqueta}: {r.get('puntuacion', '?')}/10")
        for p in r.get("problemas", []):
            print(f"   - {p}")
        print(f"   Versión final: {r.get('traduccion_corregida', '')}")
    print(f"\nConsistencia: {rev.get('consistencia', '')}")
    print(f"Veredicto: {rev.get('veredicto', '')}\n{sep}")


def configurar_consola() -> None:
    """Evita errores de codificación (acentos, →, ─) en la consola de Windows."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def main() -> None:
    configurar_consola()
    p = argparse.ArgumentParser(description="Traducción multi-agente EN→FR→ES + revisión")
    p.add_argument("texto", nargs="?", help="Texto en inglés (si se omite, se lee de stdin)")
    p.add_argument("--mock", action="store_true", help="Usar cliente simulado (sin API)")
    p.add_argument("--modelo", help=f"Modelo de DeepSeek (por defecto {DEEPSEEK_MODELO_POR_DEFECTO})")
    p.add_argument("--pensar", action="store_true", help="Activar el modo thinking de DeepSeek (más lento)")
    p.add_argument("--json", metavar="ARCHIVO", help="Guardar estado completo en JSON")
    args = p.parse_args()

    texto = args.texto
    if not texto:
        if sys.stdin.isatty():
            # Sin argumento y sin tubería: pedir el texto en vez de quedarse esperando.
            texto = input("Escribe el texto en inglés y presiona Enter:\n> ")
        else:
            texto = sys.stdin.read()  # p. ej.  Get-Content texto.txt | python ...
    if not texto or not texto.strip():
        p.error("Proporciona un texto en inglés.")

    cargar_env()
    print("Iniciando pipeline " + ("(modo simulado)" if args.mock else "con DeepSeek") + "...", flush=True)
    llm: ClienteLLM = ClienteSimulado() if args.mock else ClienteDeepSeek(args.modelo, pensar=args.pensar)
    estado = crear_pipeline(llm).ejecutar(texto)
    imprimir_resultado(estado)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(estado), f, ensure_ascii=False, indent=2)
        print(f"Estado guardado en {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
