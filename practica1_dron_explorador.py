"""
================================================================================
Laboratorio de Sistemas Multiagente - Portafolio (PIA)
Proyecto: AgroSwarm - Sistema multiagente de drones para la detección
          temprana de plagas en cultivos
Práctica 1 adaptada - Fase 1: Fundamentos de agentes
Un dron de inspección: arquitectura reactiva frente a basada en modelos
================================================================================

PROPÓSITO DEL PROGRAMA

Un dron de inspección despega de la estación base, ubicada en la esquina
(0, 0) de un cultivo dividido en parcelas, y debe encontrar los focos de
plaga antes de que se propaguen. El problema tiene cuatro características
propias del dominio agrícola, ausentes en el ejemplo genérico de las
prácticas:

    1. La plaga se propaga. Cada foco que nadie ha detectado puede
       contagiar, con cierta probabilidad por paso, a una parcela sana
       vecina. Detectar tarde significa más parcelas infestadas.

    2. Batería y regreso a base. El dron tiene una batería limitada. Si
       se agota lejos de la base, el dron se pierde. Por ello debe
       decidir en todo momento si le alcanza la batería para seguir
       inspeccionando o si debe regresar a recargar.

    3. Detectar no es eliminar. La cámara del dron ve desde el aire su
       parcela y las ocho vecinas (ventana de 3x3). Si ve una parcela con
       síntomas, la marca como sospecha; para confirmar el foco debe
       sobrevolarla. Un foco confirmado se registra en el mapa de
       infestación y se manda a fumigar, con lo que deja de propagarse.

    4. Medida de desempeño agrícola: porcentaje del cultivo inspeccionado,
       focos confirmados, parcelas que siguen infestadas sin detectar al
       final de la misión, tiempo hasta la detección y drones perdidos.

El programa compara dos programas de agente (Russell y Norvig, 2020):

    - Dron reactivo (reflejo simple): decide solo con lo que ve en este
      instante y con su posición GPS. No recuerda qué parcelas ya
      inspeccionó ni conoce el camino de regreso; para volver a la base
      se acerca "en línea recta" y puede quedar atrapado por zonas no
      sobrevolables.

    - Dron basado en modelos: mantiene un mapa interno del cultivo con lo
      que ha visto. Lo usa para tres cosas: ir a la sospecha o a la
      frontera de territorio no inspeccionado más cercana, calcular la
      ruta real de regreso a la base, y saber cuándo ya no queda nada
      por inspeccionar.

ARCHIVOS QUE GENERA

    1. Reporte en terminal: tabla comparativa del episodio, bitácora y
       experimento con varias semillas (promedios).
    2. comparacion_practica1.png: mapa del cultivo y recorrido de cada dron.
    3. simulacion_practica1.gif: animación del recorrido paso a paso.

FORMA DE EJECUCIÓN

    python practica1_dron_explorador.py
"""

from __future__ import annotations

import os
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def _ruta_junto_al_script(nombre_archivo: str) -> str:
    """Ruta absoluta de un archivo de salida, junto a este script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), nombre_archivo)


# =============================================================================
# SECCIÓN 0. PARÁMETROS CONFIGURABLES
# =============================================================================

@dataclass
class ParametrosPractica:
    # --- Cultivo ---
    tamano_cultivo: int = 12            # parcelas por lado (12 x 12)
    densidad_restringidas: float = 0.12  # fracción de parcelas no sobrevolables
    num_focos_iniciales: int = 4         # focos de plaga al inicio de la misión
    prob_propagacion: float = 0.02       # prob. por paso de que un foco no detectado contagie a una vecina
    semilla: int = 42

    # --- Dron ---
    bateria_maxima: int = 40   # pasos de vuelo con la batería llena
    tiempo_recarga: int = 5    # pasos que tarda en recargar en la base
    margen_bateria: int = 2    # reserva de seguridad para el regreso
    pasos_mision: int = 150    # duración total de la misión

    # --- Medida de desempeño ---
    peso_foco_confirmado: float = 10.0
    peso_parcela_inspeccionada: float = 1.0
    peso_parcela_infestada: float = 5.0   # se resta por cada parcela que sigue infestada al final
    peso_energia: float = 0.2             # se resta por cada paso de vuelo
    peso_dron_perdido: float = 20.0

    # --- Experimento con varias semillas ---
    num_semillas_experimento: int = 30

    ruta_figura: str = "comparacion_practica1.png"
    ruta_animacion: str = "simulacion_practica1.gif"


# =============================================================================
# SECCIÓN 1. El cultivo (entorno)
# =============================================================================

class Estado(Enum):
    """Estado REAL de una parcela. Solo el cultivo lo conoce."""
    SANA = "sana"
    RESTRINGIDA = "restringida"   # zona no sobrevolable (árboles altos, líneas eléctricas)
    PLAGA = "plaga"               # foco activo, sin detectar
    TRATADA = "tratada"           # foco confirmado y enviado a fumigar


class Observacion(Enum):
    """Lo que la cámara del dron reporta de una parcela."""
    SANA = "sana"
    RESTRINGIDA = "restringida"
    SOSPECHA = "sospecha"         # se ven síntomas desde el aire; falta confirmar
    CONFIRMADA = "confirmada"     # foco ya registrado en el mapa de infestación


BASE = (0, 0)


def vecinos4(pos):
    f, c = pos
    return [(f - 1, c), (f + 1, c), (f, c - 1), (f, c + 1)]


@dataclass
class Cultivo:
    tamano: int
    densidad_restringidas: float
    num_focos_iniciales: int
    prob_propagacion: float
    semilla: int

    terreno: list = field(init=False)
    aparicion_foco: dict = field(init=False)      # posición -> paso en que apareció el foco
    confirmacion_foco: dict = field(init=False)   # posición -> paso en que se confirmó
    inspeccionadas: set = field(init=False)       # parcelas vistas al menos una vez por la cámara

    def __post_init__(self) -> None:
        rng = random.Random(self.semilla)
        # Generador independiente para la propagación: así la dinámica de la
        # plaga es idéntica para ambas arquitecturas mientras no difieran en
        # qué focos detectan.
        self.rng_dinamica = random.Random(self.semilla + 1000)

        n = self.tamano
        self.terreno = [[Estado.SANA for _ in range(n)] for _ in range(n)]
        for f in range(n):
            for c in range(n):
                if (f, c) != BASE and abs(f) + abs(c) > 1 and rng.random() < self.densidad_restringidas:
                    self.terreno[f][c] = Estado.RESTRINGIDA

        candidatas = [
            (f, c) for f in range(n) for c in range(n)
            if self.terreno[f][c] == Estado.SANA and f + c >= 4
        ]
        rng.shuffle(candidatas)
        self.aparicion_foco = {}
        for pos in candidatas[: self.num_focos_iniciales]:
            self.terreno[pos[0]][pos[1]] = Estado.PLAGA
            self.aparicion_foco[pos] = 0
        self.confirmacion_foco = {}
        self.inspeccionadas = set()

    def dentro(self, pos) -> bool:
        return 0 <= pos[0] < self.tamano and 0 <= pos[1] < self.tamano

    def estado(self, pos) -> Estado:
        return self.terreno[pos[0]][pos[1]]

    def es_transitable(self, pos) -> bool:
        return self.dentro(pos) and self.estado(pos) != Estado.RESTRINGIDA

    def restringidas(self) -> list:
        return [(f, c) for f in range(self.tamano) for c in range(self.tamano)
                if self.terreno[f][c] == Estado.RESTRINGIDA]

    def percibir(self, pos) -> dict:
        """Ventana de 3x3 alrededor del dron: lo que ve la cámara."""
        f, c = pos
        percepto = {}
        for df in (-1, 0, 1):
            for dc in (-1, 0, 1):
                v = (f + df, c + dc)
                if self.dentro(v):
                    self.inspeccionadas.add(v)
                    percepto[v] = self._observar(v)
        return percepto

    def _observar(self, pos) -> Observacion:
        return {
            Estado.SANA: Observacion.SANA,
            Estado.RESTRINGIDA: Observacion.RESTRINGIDA,
            Estado.PLAGA: Observacion.SOSPECHA,
            Estado.TRATADA: Observacion.CONFIRMADA,
        }[self.estado(pos)]

    def sobrevolar(self, pos, paso: int) -> bool:
        """Sobrevuelo a baja altura: confirma el foco si existe. Devuelve True si confirmó uno."""
        if self.estado(pos) == Estado.PLAGA:
            self.terreno[pos[0]][pos[1]] = Estado.TRATADA
            self.confirmacion_foco[pos] = paso
            return True
        return False

    def propagar(self, paso: int) -> list:
        """Cada foco no detectado puede contagiar a una parcela sana vecina."""
        nuevos = []
        focos_activos = [p for p in self.aparicion_foco if self.estado(p) == Estado.PLAGA]
        for foco in focos_activos:
            if self.rng_dinamica.random() < self.prob_propagacion:
                sanas = [v for v in vecinos4(foco)
                         if self.dentro(v) and v != BASE and self.estado(v) == Estado.SANA]
                if sanas:
                    nuevo = self.rng_dinamica.choice(sanas)
                    self.terreno[nuevo[0]][nuevo[1]] = Estado.PLAGA
                    self.aparicion_foco[nuevo] = paso
                    nuevos.append(nuevo)
        return nuevos

    def parcelas_cultivables(self) -> int:
        return sum(1 for f in range(self.tamano) for c in range(self.tamano)
                   if self.terreno[f][c] != Estado.RESTRINGIDA)

    def infestadas_sin_detectar(self) -> int:
        return sum(1 for p in self.aparicion_foco if self.estado(p) == Estado.PLAGA)


# =============================================================================
# SECCIÓN 2. Medida de desempeño
# =============================================================================

@dataclass
class Desempeno:
    pasos_vuelo: int = 0
    focos_confirmados: int = 0
    revisitas: int = 0          # movimientos hacia una parcela ya sobrevolada
    regresos_a_base: int = 0
    perdido: bool = False       # se quedó sin batería fuera de la base


def resumir_episodio(cultivo: Cultivo, desempenos: list, p: ParametrosPractica) -> dict:
    """Métricas a nivel de misión, comunes a todas las arquitecturas."""
    cultivables = cultivo.parcelas_cultivables()
    inspeccionadas = sum(1 for x in cultivo.inspeccionadas if cultivo.estado(x) != Estado.RESTRINGIDA)
    confirmados = len(cultivo.confirmacion_foco)
    tiempos = [cultivo.confirmacion_foco[x] - cultivo.aparicion_foco[x] for x in cultivo.confirmacion_foco]
    infestadas = cultivo.infestadas_sin_detectar()
    energia = sum(d.pasos_vuelo for d in desempenos)
    perdidos = sum(1 for d in desempenos if d.perdido)
    puntaje = (
        confirmados * p.peso_foco_confirmado
        + inspeccionadas * p.peso_parcela_inspeccionada
        - infestadas * p.peso_parcela_infestada
        - energia * p.peso_energia
        - perdidos * p.peso_dron_perdido
    )
    return {
        "cobertura_pct": 100.0 * inspeccionadas / cultivables,
        "parcelas_inspeccionadas": inspeccionadas,
        "focos_totales": len(cultivo.aparicion_foco),
        "focos_confirmados": confirmados,
        "infestadas_sin_detectar": infestadas,
        "tiempo_deteccion": statistics.mean(tiempos) if tiempos else float("nan"),
        "energia": energia,
        "revisitas": sum(d.revisitas for d in desempenos),
        "regresos_a_base": sum(d.regresos_a_base for d in desempenos),
        "drones_perdidos": perdidos,
        "puntaje": puntaje,
    }


# =============================================================================
# SECCIÓN 3. Clase base del dron: ciclo percepción-decisión-acción
# =============================================================================

def direccion(origen, destino) -> str:
    f, c = origen
    df, dc = destino
    if df < f:
        return "arriba"
    if df > f:
        return "abajo"
    if dc < c:
        return "izquierda"
    if dc > c:
        return "derecha"
    return "esperar"


def destino_de(pos, accion):
    f, c = pos
    return {"arriba": (f - 1, c), "abajo": (f + 1, c), "izquierda": (f, c - 1),
            "derecha": (f, c + 1)}.get(accion, pos)


class Dron:
    """
    Estructura común a ambas arquitecturas: percibir, decidir, actuar.
    Las subclases solo redefinen decidir() (y el basado en modelos
    también percibir(), para actualizar su mapa).

    Estados operativos:
      "volando"     - en misión
      "recargando"  - en la base, cargando batería
      "terminado"   - aterrizó porque ya no le queda nada por inspeccionar
      "perdido"     - se quedó sin batería fuera de la base
    """

    def __init__(self, p: ParametrosPractica, rng: random.Random):
        self.p = p
        self.rng = rng
        self.posicion = BASE
        self.bateria = p.bateria_maxima
        self.estado = "volando"
        self.recarga_restante = 0
        self.regresando = False
        self.sobrevoladas = {BASE}
        self.desempeno = Desempeno()

    def percibir(self, cultivo: Cultivo) -> dict:
        return cultivo.percibir(self.posicion)

    def decidir(self, percepto: dict) -> tuple[str, str]:
        raise NotImplementedError

    def actuar(self, accion: str, cultivo: Cultivo, paso: int) -> bool:
        """Mueve el dron, descuenta batería y confirma focos. True si confirmó uno."""
        if accion == "aterrizar":
            self.estado = "terminado"
            return False

        destino = destino_de(self.posicion, accion)
        if accion != "esperar" and cultivo.es_transitable(destino):
            if destino in self.sobrevoladas:
                self.desempeno.revisitas += 1
            self.posicion = destino
            self.sobrevoladas.add(destino)
        self.bateria -= 1
        self.desempeno.pasos_vuelo += 1

        confirmo = cultivo.sobrevolar(self.posicion, paso)
        if confirmo:
            self.desempeno.focos_confirmados += 1

        if self.posicion == BASE and self.regresando:
            self.regresando = False
            self.estado = "recargando"
            self.recarga_restante = self.p.tiempo_recarga
            self.desempeno.regresos_a_base += 1
        elif self.bateria <= 0 and self.posicion != BASE:
            self.estado = "perdido"
            self.desempeno.perdido = True
        return confirmo

    def paso_recarga(self) -> None:
        self.recarga_restante -= 1
        if self.recarga_restante <= 0:
            self.bateria = self.p.bateria_maxima
            self.estado = "volando"


# =============================================================================
# SECCIÓN 4. Arquitectura 1: dron reactivo (reflejo simple)
# =============================================================================

class DronReactivo(Dron):
    """
    Reglas condición-acción sobre el percepto actual y la lectura del GPS.
    No guarda mapa: no sabe qué parcelas ya revisó ni cuál es el camino
    real de regreso a la base.
    """

    def decidir(self, percepto: dict) -> tuple[str, str]:
        f, c = self.posicion
        distancia_recta = f + c   # distancia Manhattan a la base, leída del GPS

        # Regla 1: batería baja -> acercarse a la base "en línea recta".
        if self.bateria <= distancia_recta + self.p.margen_bateria:
            self.regresando = True
            mas_cerca = [v for v in vecinos4(self.posicion)
                         if percepto.get(v) not in (None, Observacion.RESTRINGIDA)
                         and v[0] + v[1] < distancia_recta]
            if mas_cerca:
                return direccion(self.posicion, self.rng.choice(mas_cerca)), \
                    "Batería baja: se acerca a la base en línea recta según el GPS."
            libres = [v for v in vecinos4(self.posicion)
                      if percepto.get(v) not in (None, Observacion.RESTRINGIDA)]
            if libres:
                return direccion(self.posicion, self.rng.choice(libres)), \
                    "Batería baja, pero una zona no sobrevolable le bloquea el paso directo; se mueve al azar."
            return "esperar", "Batería baja y sin salida; permanece en su lugar."

        # Regla 2: sospecha de plaga en una parcela contigua -> ir a confirmarla.
        sospechas = [v for v in vecinos4(self.posicion) if percepto.get(v) == Observacion.SOSPECHA]
        if sospechas:
            return direccion(self.posicion, sospechas[0]), \
                f"Ve síntomas en {sospechas[0]}; la sobrevuela para confirmar el foco."

        # Regla 3: moverse al azar a una parcela sobrevolable.
        libres = [v for v in vecinos4(self.posicion)
                  if percepto.get(v) not in (None, Observacion.RESTRINGIDA)]
        if libres:
            destino = self.rng.choice(libres)
            return direccion(self.posicion, destino), \
                f"No ve síntomas cercanos; elige al azar {destino}, sin saber si ya la inspeccionó."
        return "esperar", "No hay parcelas sobrevolables alrededor; permanece en su lugar."


# =============================================================================
# SECCIÓN 5. Arquitectura 2: dron basado en modelos
# =============================================================================

class DronBasadoEnModelos(Dron):
    """
    Agrega un estado interno: mapa_interno, con la última observación de
    cada parcela vista. Con él calcula distancias reales (BFS sobre
    parcelas conocidas), elige objetivos y planea el regreso.
    """

    def __init__(self, p: ParametrosPractica, rng: random.Random):
        super().__init__(p, rng)
        self.mapa_interno: dict = {BASE: Observacion.SANA}

    def percibir(self, cultivo: Cultivo) -> dict:
        percepto = super().percibir(cultivo)
        self.mapa_interno.update(percepto)
        return percepto

    # --- utilidades sobre el mapa interno ---

    def _transitable_conocida(self, pos) -> bool:
        obs = self.mapa_interno.get(pos)
        return obs is not None and obs != Observacion.RESTRINGIDA

    def _bfs(self, origen) -> dict:
        """Distancias desde origen a toda parcela conocida y sobrevolable."""
        dist = {origen: 0}
        cola = deque([origen])
        while cola:
            actual = cola.popleft()
            for v in vecinos4(actual):
                if v not in dist and self._transitable_conocida(v):
                    dist[v] = dist[actual] + 1
                    cola.append(v)
        return dist

    def _primer_paso(self, objetivo) -> str:
        padres = {self.posicion: None}
        cola = deque([self.posicion])
        while cola:
            actual = cola.popleft()
            if actual == objetivo:
                break
            for v in vecinos4(actual):
                if v not in padres and self._transitable_conocida(v):
                    padres[v] = actual
                    cola.append(v)
        if objetivo not in padres or objetivo == self.posicion:
            return "esperar"
        paso = objetivo
        while padres[paso] != self.posicion:
            paso = padres[paso]
        return direccion(self.posicion, paso)

    def _es_frontera(self, pos) -> bool:
        tam = self.p.tamano_cultivo
        return any(0 <= v[0] < tam and 0 <= v[1] < tam and v not in self.mapa_interno
                   for v in vecinos4(pos))

    def _objetivos(self) -> list:
        """Sospechas pendientes primero; después, fronteras de lo no inspeccionado."""
        sospechas = [x for x, o in self.mapa_interno.items() if o == Observacion.SOSPECHA]
        fronteras = [x for x in self.mapa_interno
                     if self._transitable_conocida(x) and self._es_frontera(x)]
        return [("sospecha", sospechas), ("frontera", fronteras)]

    def decidir(self, percepto: dict) -> tuple[str, str]:
        dist_aqui = self._bfs(self.posicion)
        dist_base = self._bfs(BASE)
        regreso = dist_base.get(self.posicion, 0)

        # Regla 1: la batería apenas alcanza para volver por la ruta conocida.
        if self.posicion != BASE and self.bateria <= regreso + self.p.margen_bateria:
            self.regresando = True
            return self._primer_paso(BASE), \
                f"Batería {self.bateria}: la ruta conocida a la base mide {regreso}; regresa a recargar."

        # Regla 2: ir al objetivo más cercano que sea alcanzable con la
        # batería disponible, contando el viaje de regreso a la base.
        for tipo, candidatos in self._objetivos():
            factibles = [
                x for x in candidatos
                if x in dist_aqui and x in dist_base
                and dist_aqui[x] + dist_base[x] <= self.bateria - self.p.margen_bateria
            ]
            if factibles:
                objetivo = min(factibles, key=lambda x: (dist_aqui[x], x))
                if tipo == "sospecha":
                    motivo = f"Su mapa registra una sospecha en {objetivo} a {dist_aqui[objetivo]} pasos; va a confirmarla."
                else:
                    motivo = f"Va a la frontera de lo no inspeccionado más cercana, {objetivo}."
                return self._primer_paso(objetivo), motivo

        # Regla 3: nada alcanzable con la batería actual.
        hay_pendientes = any(c for _, c in self._objetivos())
        if self.posicion != BASE:
            self.regresando = True
            return self._primer_paso(BASE), "Ningún objetivo alcanzable con la batería actual; regresa a la base."
        if hay_pendientes and self.bateria < self.p.bateria_maxima:
            self.regresando = True
            return "esperar", "En la base con batería parcial y objetivos lejanos; se queda a recargar."
        return "aterrizar", "Su mapa no tiene sospechas ni fronteras alcanzables: misión de inspección terminada."


# =============================================================================
# SECCIÓN 6. Simulación de una misión
# =============================================================================

def ejecutar_mision(clase_dron, p: ParametrosPractica) -> dict:
    cultivo = Cultivo(p.tamano_cultivo, p.densidad_restringidas, p.num_focos_iniciales,
                      p.prob_propagacion, p.semilla)
    restringidas = cultivo.restringidas()
    dron = clase_dron(p, random.Random(p.semilla))
    trayectoria = [dron.posicion]
    bitacora = []

    for paso in range(1, p.pasos_mision + 1):
        for nuevo in cultivo.propagar(paso):
            bitacora.append({"paso": paso, "tipo": "propagacion",
                             "texto": f"La plaga se propaga a la parcela {nuevo}."})

        if dron.estado == "recargando":
            dron.paso_recarga()
        elif dron.estado == "volando":
            percepto = dron.percibir(cultivo)
            accion, motivo = dron.decidir(percepto)
            if accion == "esperar" and dron.posicion == BASE and dron.regresando:
                # Se queda en la base a recargar sin consumir batería.
                dron.regresando = False
                dron.estado = "recargando"
                dron.recarga_restante = p.tiempo_recarga
                dron.desempeno.regresos_a_base += 1
                confirmo = False
            else:
                confirmo = dron.actuar(accion, cultivo, paso)
            bitacora.append({"paso": paso, "tipo": "decision", "texto": motivo})
            if confirmo:
                bitacora.append({"paso": paso, "tipo": "confirmacion",
                                 "texto": f"Foco confirmado en {dron.posicion} y enviado a fumigar."})
            if dron.estado == "perdido":
                bitacora.append({"paso": paso, "tipo": "perdido",
                                 "texto": f"El dron se queda sin batería en {dron.posicion} y se pierde."})
            if dron.estado == "terminado":
                bitacora.append({"paso": paso, "tipo": "terminado",
                                 "texto": "El dron aterriza: ya inspeccionó todo lo alcanzable."})
        trayectoria.append(dron.posicion)

    resumen = resumir_episodio(cultivo, [dron.desempeno], p)
    return {
        "resumen": resumen,
        "trayectoria": trayectoria,
        "bitacora": bitacora,
        "restringidas": restringidas,
        "aparicion_foco": dict(cultivo.aparicion_foco),
        "confirmacion_foco": dict(cultivo.confirmacion_foco),
    }


ARQUITECTURAS = [("Reactivo", DronReactivo), ("Basado en modelos", DronBasadoEnModelos)]


def comparar_arquitecturas(p: ParametrosPractica) -> dict:
    return {nombre: ejecutar_mision(clase, p) for nombre, clase in ARQUITECTURAS}


def experimento_multisemilla(p: ParametrosPractica) -> dict:
    """Repite la comparación con varias semillas para no depender de un solo mapa."""
    from dataclasses import replace
    datos = {nombre: [] for nombre, _ in ARQUITECTURAS}
    for s in range(1, p.num_semillas_experimento + 1):
        ps = replace(p, semilla=s)
        for nombre, clase in ARQUITECTURAS:
            datos[nombre].append(ejecutar_mision(clase, ps)["resumen"])
    return datos


# =============================================================================
# SECCIÓN 7. Reporte en terminal
# =============================================================================

METRICAS = [
    ("cobertura_pct", "Cultivo inspeccionado (%)"),
    ("focos_totales", "Focos totales (iniciales + propagados)"),
    ("focos_confirmados", "Focos confirmados"),
    ("infestadas_sin_detectar", "Parcelas infestadas sin detectar"),
    ("tiempo_deteccion", "Tiempo medio hasta detección (pasos)"),
    ("energia", "Pasos de vuelo (energía)"),
    ("revisitas", "Revisitas"),
    ("regresos_a_base", "Regresos a base"),
    ("drones_perdidos", "Drones perdidos"),
    ("puntaje", "Puntaje de desempeño"),
]


def _fmt(x) -> str:
    if isinstance(x, float):
        return "n/d" if x != x else f"{x:.1f}"
    return str(x)


def imprimir_comparacion(resultados: dict, p: ParametrosPractica) -> None:
    print("=" * 80)
    print("AgroSwarm - PRÁCTICA 1: UN DRON, REACTIVO VS. BASADO EN MODELOS")
    print(f"Cultivo {p.tamano_cultivo}x{p.tamano_cultivo} | Zonas restringidas {p.densidad_restringidas:.0%} | "
          f"Focos iniciales {p.num_focos_iniciales} | Propagación {p.prob_propagacion:.0%}/paso")
    print(f"Batería {p.bateria_maxima} | Recarga {p.tiempo_recarga} | Misión {p.pasos_mision} pasos | Semilla {p.semilla}")
    print("=" * 80)
    nombres = list(resultados)
    print(f"{'Métrica':<42}{nombres[0]:>16}{nombres[1]:>22}")
    print("-" * 80)
    for clave, etiqueta in METRICAS:
        print(f"{etiqueta:<42}{_fmt(resultados[nombres[0]]['resumen'][clave]):>16}"
              f"{_fmt(resultados[nombres[1]]['resumen'][clave]):>22}")
    print("=" * 80)


def imprimir_bitacora(resultados: dict, max_decisiones: int = 4) -> None:
    print("\nBITÁCORA (eventos relevantes)")
    print("-" * 80)
    for nombre, datos in resultados.items():
        print(f"\n[{nombre}]")
        decisiones = [e for e in datos["bitacora"] if e["tipo"] == "decision"][:max_decisiones]
        for e in decisiones:
            print(f"  Paso {e['paso']:>3}: {e['texto']}")
        relevantes = [e for e in datos["bitacora"] if e["tipo"] in ("confirmacion", "perdido", "terminado")]
        relevantes += [e for e in datos["bitacora"] if e["tipo"] == "decision" and "regresa" in e["texto"]][:2]
        relevantes.sort(key=lambda e: e["paso"])
        if relevantes:
            print("  (...)")
            for e in relevantes:
                print(f"  Paso {e['paso']:>3}: {e['texto']}")


def imprimir_experimento(datos: dict, p: ParametrosPractica) -> None:
    print("\n" + "=" * 80)
    print(f"EXPERIMENTO CON {p.num_semillas_experimento} SEMILLAS (promedio ± desviación estándar)")
    print("=" * 80)
    nombres = list(datos)
    print(f"{'Métrica':<42}{nombres[0]:>16}{nombres[1]:>22}")
    print("-" * 80)
    for clave, etiqueta in METRICAS:
        celdas = []
        for n in nombres:
            valores = [r[clave] for r in datos[n] if r[clave] == r[clave]]
            if valores:
                celdas.append(f"{statistics.mean(valores):.1f} ± {statistics.pstdev(valores):.1f}")
            else:
                celdas.append("n/d")
        print(f"{etiqueta:<42}{celdas[0]:>16}{celdas[1]:>22}")
    gana = sum(1 for a, b in zip(datos[nombres[0]], datos[nombres[1]]) if b["puntaje"] > a["puntaje"])
    print("-" * 80)
    print(f"El dron basado en modelos obtuvo mayor puntaje en {gana} de {p.num_semillas_experimento} cultivos.")
    print("=" * 80)


# =============================================================================
# SECCIÓN 8. Figura comparativa
# =============================================================================

def _preparar_eje(eje, tamano, titulo):
    eje.set_title(titulo, fontsize=13, fontweight="bold", pad=34)
    eje.set_xlim(-0.5, tamano - 0.5)
    eje.set_ylim(tamano - 0.5, -0.5)
    eje.set_aspect("equal")
    eje.set_xticks([])
    eje.set_yticks([])
    eje.set_xticks([x - 0.5 for x in range(tamano + 1)], minor=True)
    eje.set_yticks([y - 0.5 for y in range(tamano + 1)], minor=True)
    eje.grid(which="minor", color="#D1D5DB", linewidth=0.6)
    eje.tick_params(which="minor", bottom=False, left=False)
    eje.set_facecolor("#ECFCCB")   # verde claro: parcelas de cultivo


def generar_figura(resultados: dict, p: ParametrosPractica) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("\nNota: matplotlib no está disponible; no se generó la figura.")
        return

    fig, ejes = plt.subplots(1, 2, figsize=(13, 6.8))
    for eje, (nombre, datos) in zip(ejes, resultados.items()):
        _preparar_eje(eje, p.tamano_cultivo, nombre)
        if datos["restringidas"]:
            eje.scatter([x[1] for x in datos["restringidas"]], [x[0] for x in datos["restringidas"]],
                        marker="s", s=420, color="#4B5563", zorder=2)
        tr = datos["trayectoria"]
        eje.plot([x[1] for x in tr], [x[0] for x in tr], "-", color="#1D4ED8", alpha=0.45, linewidth=1.2, zorder=3)
        for pos, paso in datos["aparicion_foco"].items():
            confirmado = pos in datos["confirmacion_foco"]
            eje.scatter(pos[1], pos[0], marker="*" if paso == 0 else "D", s=330 if paso == 0 else 120,
                        color="#F59E0B" if confirmado else "#DC2626", edgecolors="#7F1D1D",
                        linewidths=1.0, zorder=5)
        eje.scatter(0, 0, marker="s", s=300, color="#16A34A", edgecolors="white", linewidths=1.5, zorder=6)
        r = datos["resumen"]
        eje.text(0.5, 1.01,
                 f"Inspeccionado: {r['cobertura_pct']:.0f}%   Focos: {r['focos_confirmados']}/{r['focos_totales']}   "
                 f"Infestadas sin detectar: {r['infestadas_sin_detectar']}\n"
                 f"Pasos de vuelo: {r['energia']}   Perdidos: {r['drones_perdidos']}   Puntaje: {r['puntaje']:.1f}",
                 transform=eje.transAxes, ha="center", va="bottom", fontsize=8.5, family="monospace")

    leyenda = [
        Patch(facecolor="#4B5563", label="Zona no sobrevolable"),
        Patch(facecolor="#16A34A", label="Estación base"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#F59E0B", markeredgecolor="#7F1D1D", markersize=14, label="Foco inicial confirmado"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#DC2626", markeredgecolor="#7F1D1D", markersize=14, label="Foco inicial sin detectar"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#F59E0B", markeredgecolor="#7F1D1D", markersize=8, label="Foco propagado confirmado"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#DC2626", markeredgecolor="#7F1D1D", markersize=8, label="Foco propagado sin detectar"),
        Line2D([0], [0], color="#1D4ED8", label="Recorrido del dron"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("AgroSwarm - Práctica 1: dron reactivo frente a dron basado en modelos", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    ruta = _ruta_junto_al_script(p.ruta_figura)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura generada: {ruta}")


# =============================================================================
# SECCIÓN 9. Animación
# =============================================================================

def generar_animacion(resultados: dict, p: ParametrosPractica, fps: int = 10) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import PIL  # noqa: F401
    except ImportError:
        print("\nNota: falta matplotlib o Pillow (pip install pillow); no se generó la animación.")
        return

    fig, ejes = plt.subplots(1, 2, figsize=(13, 6.8))
    refs = {}
    for eje, (nombre, datos) in zip(ejes, resultados.items()):
        _preparar_eje(eje, p.tamano_cultivo, nombre)
        if datos["restringidas"]:
            eje.scatter([x[1] for x in datos["restringidas"]], [x[0] for x in datos["restringidas"]],
                        marker="s", s=420, color="#4B5563", zorder=2)
        eje.scatter(0, 0, marker="s", s=300, color="#16A34A", zorder=2)
        focos = list(datos["aparicion_foco"])
        sc_focos = eje.scatter([x[1] for x in focos], [x[0] for x in focos], marker="*", s=300,
                               edgecolors="#7F1D1D", linewidths=1.0, zorder=5)
        (linea,) = eje.plot([], [], "-", color="#1D4ED8", alpha=0.5, zorder=3)
        dron = eje.scatter([], [], s=200, color="#1D4ED8", edgecolors="white", linewidths=1.5, zorder=6)
        texto = eje.text(0.5, 1.01, "", transform=eje.transAxes, ha="center", va="bottom",
                         fontsize=9, family="monospace")
        refs[nombre] = (sc_focos, focos, linea, dron, texto)
    fig.suptitle("AgroSwarm - Práctica 1: recorrido paso a paso (rojo: foco activo, naranja: confirmado)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def actualizar(t):
        artistas = []
        for nombre, datos in resultados.items():
            sc_focos, focos, linea, dron, texto = refs[nombre]
            tr = datos["trayectoria"][: t + 1]
            linea.set_data([x[1] for x in tr], [x[0] for x in tr])
            dron.set_offsets([[tr[-1][1], tr[-1][0]]])
            colores = []
            for f in focos:
                if datos["aparicion_foco"][f] > t:
                    colores.append((0, 0, 0, 0))
                elif datos["confirmacion_foco"].get(f, 10**9) <= t:
                    colores.append("#F59E0B")
                else:
                    colores.append("#DC2626")
            sc_focos.set_facecolor(colores)
            sc_focos.set_edgecolor(["#7F1D1D" if c != (0, 0, 0, 0) else (0, 0, 0, 0) for c in colores])
            activos = sum(1 for c in colores if c == "#DC2626")
            texto.set_text(f"Paso {t}/{p.pasos_mision}   Focos activos: {activos}")
            artistas += [linea, dron, sc_focos, texto]
        return artistas

    anim = animation.FuncAnimation(fig, actualizar, frames=p.pasos_mision + 1, interval=1000 / fps)
    ruta = _ruta_junto_al_script(p.ruta_animacion)
    try:
        anim.save(ruta, writer=animation.PillowWriter(fps=fps))
        print(f"Animación generada: {ruta}")
    finally:
        plt.close(fig)


# =============================================================================
# SECCIÓN 10. Punto de entrada
# =============================================================================

if __name__ == "__main__":
    parametros = ParametrosPractica()
    resultados = comparar_arquitecturas(parametros)
    imprimir_comparacion(resultados, parametros)
    imprimir_bitacora(resultados)
    imprimir_experimento(experimento_multisemilla(parametros), parametros)
    generar_figura(resultados, parametros)
    generar_animacion(resultados, parametros)
