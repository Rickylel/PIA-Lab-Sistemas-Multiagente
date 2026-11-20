"""
================================================================================
Laboratorio de Sistemas Multiagente - Portafolio (PIA)
Proyecto: AgroSwarm - Sistema multiagente de drones para la detección
          temprana de plagas en cultivos
Práctica 2 adaptada - Fase 2: Coordinación y comunicación
Flota de drones: pizarrón compartido (estación base) frente a mensajería
================================================================================

PROPÓSITO DEL PROGRAMA

Este programa continúa la Práctica 1. Ahora una FLOTA de drones despega de
la misma estación base para inspeccionar el cultivo. El mundo es el mismo
de la Práctica 1: zonas no sobrevolables, plaga que se propaga, batería
limitada con regreso obligatorio a la base, y focos que se confirman
sobrevolándolos.

Sin coordinación, todos los drones saldrían juntos de la base hacia la
misma frontera y revisarían las mismas parcelas. El programa compara dos
mecanismos para repartir el trabajo:

    1. Pizarrón compartido: la estación base mantiene un único mapa de
       infestación y un registro de reservas ("qué dron va hacia qué
       parcela"). Cada dron lo lee y lo actualiza al instante.

    2. Mensajería directa (inspirada en FIPA-ACL): no hay mapa central.
       Cada dron tiene su propio mapa y lo actualiza con lo que ve y con
       los avisos por radio que le mandan los demás. Los avisos tardan
       "retraso_mensaje" pasos en llegar, así que el conocimiento de cada
       dron sobre el equipo siempre está algo desactualizado. Si dos drones
       reservan el mismo objetivo en el mismo paso, hay una disputa que se
       resuelve con un mensaje "proponer" (gana el identificador menor).

Además, el programa permite elegir la arquitectura de cada dron
(parámetro "arquitectura"): "basado_en_modelos" (la elegida en la
Práctica 1) o "reactiva". El experimento final cruza ambas arquitecturas
con ambos mecanismos para respaldar las dos decisiones de diseño con datos.

COORDINACIÓN POR RESERVAS

Con la arquitectura basada en modelos, cada dron anuncia a qué parcela
se dirige (su reserva). Los demás descartan esa parcela, y las fronteras
a menos de "radio_reserva" parcelas de ella, al elegir su propio objetivo.
Así la flota se dispersa por el cultivo en lugar de volar en grupo.

ARCHIVOS QUE GENERA

    1. Reporte en terminal: tabla comparativa, bitácora, ejemplos de
       mensajes y experimento con varias semillas.
    2. comparacion_practica2.png y simulacion_practica2.gif.

FORMA DE EJECUCIÓN

    python practica2_coordinacion_drones.py
"""

from __future__ import annotations

import os
import random
import statistics
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


def _ruta_junto_al_script(nombre_archivo: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), nombre_archivo)


# =============================================================================
# SECCIÓN 0. PARÁMETROS CONFIGURABLES
# =============================================================================

@dataclass
class ParametrosPractica:
    # --- Cultivo (igual que en la Práctica 1) ---
    tamano_cultivo: int = 12
    densidad_restringidas: float = 0.12
    num_focos_iniciales: int = 4
    prob_propagacion: float = 0.02
    semilla: int = 42

    # --- Drones ---
    num_drones: int = 3
    arquitectura: str = "basado_en_modelos"   # o "reactiva"
    bateria_maxima: int = 40
    tiempo_recarga: int = 5
    margen_bateria: int = 2
    pasos_mision: int = 150

    # --- Coordinación ---
    retraso_mensaje: int = 1   # pasos que tarda un aviso por radio (solo mensajería)
    radio_reserva: int = 2     # distancia mínima entre la frontera elegida y las reservas ajenas

    # --- Medida de desempeño del EQUIPO ---
    peso_foco_confirmado: float = 10.0
    peso_parcela_inspeccionada: float = 1.0
    peso_parcela_infestada: float = 5.0
    peso_energia: float = 0.2
    peso_dron_perdido: float = 20.0
    peso_redundancia: float = 0.5   # se resta por cada sobrevuelo de una parcela que el equipo ya había sobrevolado

    num_semillas_experimento: int = 30
    ruta_figura: str = "comparacion_practica2.png"
    ruta_animacion: str = "simulacion_practica2.gif"


# =============================================================================
# SECCIÓN 1. El cultivo (idéntico a la Práctica 1)
# =============================================================================

class Estado(Enum):
    SANA = "sana"
    RESTRINGIDA = "restringida"
    PLAGA = "plaga"
    TRATADA = "tratada"


class Observacion(Enum):
    SANA = "sana"
    RESTRINGIDA = "restringida"
    SOSPECHA = "sospecha"
    CONFIRMADA = "confirmada"


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
    aparicion_foco: dict = field(init=False)
    confirmacion_foco: dict = field(init=False)
    inspeccionadas: set = field(init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.semilla)
        self.rng_dinamica = random.Random(self.semilla + 1000)
        n = self.tamano
        self.terreno = [[Estado.SANA for _ in range(n)] for _ in range(n)]
        for f in range(n):
            for c in range(n):
                if (f, c) != BASE and abs(f) + abs(c) > 1 and rng.random() < self.densidad_restringidas:
                    self.terreno[f][c] = Estado.RESTRINGIDA
        candidatas = [(f, c) for f in range(n) for c in range(n)
                      if self.terreno[f][c] == Estado.SANA and f + c >= 4]
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
        return {Estado.SANA: Observacion.SANA, Estado.RESTRINGIDA: Observacion.RESTRINGIDA,
                Estado.PLAGA: Observacion.SOSPECHA, Estado.TRATADA: Observacion.CONFIRMADA}[self.estado(pos)]

    def sobrevolar(self, pos, paso: int) -> bool:
        if self.estado(pos) == Estado.PLAGA:
            self.terreno[pos[0]][pos[1]] = Estado.TRATADA
            self.confirmacion_foco[pos] = paso
            return True
        return False

    def propagar(self, paso: int) -> list:
        nuevos = []
        for foco in [p for p in self.aparicion_foco if self.estado(p) == Estado.PLAGA]:
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
# SECCIÓN 2. Conocimiento del equipo y mensajes
# =============================================================================

@dataclass
class Conocimiento:
    """
    Lo que un dron cree saber del cultivo y de la flota. Con pizarrón
    existe UNA sola instancia, guardada en la estación base y compartida
    por todos. Con mensajería, cada dron tiene la suya.
    """
    mapa: dict = field(default_factory=lambda: {BASE: Observacion.SANA})
    sobrevoladas: set = field(default_factory=lambda: {BASE})
    reservas: dict = field(default_factory=dict)   # id_dron -> (parcela reservada o None, paso)

    def incorporar(self, celdas: dict) -> None:
        """Agrega observaciones. Un foco confirmado nunca se sobrescribe."""
        for pos, obs in celdas.items():
            if self.mapa.get(pos) != Observacion.CONFIRMADA:
                self.mapa[pos] = obs


@dataclass
class Mensaje:
    """
    Estructura simplificada de FIPA-ACL. Performativos:
      "informar": el emisor comparte observaciones nuevas, su posición y
                  su reserva actual (aviso general a la flota).
      "proponer": el ganador de una disputa le propone al perdedor que
                  ceda la reserva.
    """
    emisor: int
    receptor: Optional[int]
    performativo: str
    contenido: dict
    paso_envio: int
    paso_entrega: int


@dataclass
class Desempeno:
    pasos_vuelo: int = 0
    focos_confirmados: int = 0
    revisitas: int = 0
    regresos_a_base: int = 0
    perdido: bool = False


# =============================================================================
# SECCIÓN 3. El dron: ciclo percepción-decisión-acción
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
    def __init__(self, id_dron: int, p: ParametrosPractica, rng: random.Random):
        self.id = id_dron
        self.p = p
        self.rng = rng
        self.posicion = BASE
        self.bateria = p.bateria_maxima
        self.estado = "volando"
        self.recarga_restante = 0
        self.regresando = False
        self.reserva = None
        self.desempeno = Desempeno()

    # --- percibir ---------------------------------------------------------
    def percibir(self, cultivo: Cultivo, conoc: Conocimiento) -> tuple[dict, dict]:
        """Devuelve el percepto y las observaciones que cambiaron el mapa."""
        percepto = cultivo.percibir(self.posicion)
        cambios = {x: o for x, o in percepto.items() if conoc.mapa.get(x) != o}
        conoc.incorporar(percepto)
        return percepto, cambios

    # --- decidir ----------------------------------------------------------
    def decidir(self, percepto: dict, conoc: Conocimiento) -> tuple[str, Optional[tuple], str]:
        if self.p.arquitectura == "reactiva":
            return self._decidir_reactivo(percepto, conoc)
        return self._decidir_modelos(conoc)

    def _reservas_ajenas(self, conoc: Conocimiento) -> list:
        return [pos for i, (pos, _) in conoc.reservas.items() if i != self.id and pos is not None]

    def _decidir_reactivo(self, percepto, conoc):
        """Reglas de la Práctica 1 + preferencia por parcelas que el equipo no ha sobrevolado."""
        f, c = self.posicion
        distancia_recta = f + c
        sobrevolables = [v for v in vecinos4(self.posicion)
                         if percepto.get(v) not in (None, Observacion.RESTRINGIDA)]
        if self.bateria <= distancia_recta + self.p.margen_bateria:
            self.regresando = True
            mas_cerca = [v for v in sobrevolables if v[0] + v[1] < distancia_recta]
            opciones = mas_cerca or sobrevolables
            if opciones:
                return direccion(self.posicion, self.rng.choice(opciones)), None, \
                    "Batería baja: se acerca a la base según el GPS."
            return "esperar", None, "Batería baja y sin salida."
        ajenas = self._reservas_ajenas(conoc)
        sospechas = [v for v in vecinos4(self.posicion)
                     if percepto.get(v) == Observacion.SOSPECHA and v not in ajenas]
        if sospechas:
            return direccion(self.posicion, sospechas[0]), sospechas[0], \
                f"Ve síntomas en {sospechas[0]} y nadie la ha reservado; va a confirmarla."
        nuevas = [v for v in sobrevolables if v not in conoc.sobrevoladas]
        if nuevas:
            destino = self.rng.choice(nuevas)
            return direccion(self.posicion, destino), None, \
                f"Elige {destino}, que el equipo no ha sobrevolado según lo que sabe."
        if sobrevolables:
            destino = self.rng.choice(sobrevolables)
            return direccion(self.posicion, destino), None, \
                f"Todo alrededor ya fue sobrevolado; elige al azar {destino}."
        return "esperar", None, "No hay parcelas sobrevolables alrededor."

    # --- utilidades del modelo ---
    def _transitable(self, conoc, pos) -> bool:
        o = conoc.mapa.get(pos)
        return o is not None and o != Observacion.RESTRINGIDA

    def _bfs(self, conoc, origen) -> dict:
        dist = {origen: 0}
        cola = deque([origen])
        while cola:
            a = cola.popleft()
            for v in vecinos4(a):
                if v not in dist and self._transitable(conoc, v):
                    dist[v] = dist[a] + 1
                    cola.append(v)
        return dist

    def _primer_paso(self, conoc, objetivo) -> str:
        padres = {self.posicion: None}
        cola = deque([self.posicion])
        while cola:
            a = cola.popleft()
            if a == objetivo:
                break
            for v in vecinos4(a):
                if v not in padres and self._transitable(conoc, v):
                    padres[v] = a
                    cola.append(v)
        if objetivo not in padres or objetivo == self.posicion:
            return "esperar"
        x = objetivo
        while padres[x] != self.posicion:
            x = padres[x]
        return direccion(self.posicion, x)

    def _es_frontera(self, conoc, pos) -> bool:
        t = self.p.tamano_cultivo
        return any(0 <= v[0] < t and 0 <= v[1] < t and v not in conoc.mapa for v in vecinos4(pos))

    def _decidir_modelos(self, conoc):
        """Arquitectura basada en modelos (Práctica 1) + reservas del equipo."""
        dist_aqui = self._bfs(conoc, self.posicion)
        dist_base = self._bfs(conoc, BASE)
        regreso = dist_base.get(self.posicion, 0)

        if self.posicion != BASE and self.bateria <= regreso + self.p.margen_bateria:
            self.regresando = True
            return self._primer_paso(conoc, BASE), None, \
                f"Batería {self.bateria}: la ruta conocida a la base mide {regreso}; regresa."

        ajenas = self._reservas_ajenas(conoc)
        prioritarias = [pos for i, (pos, _) in conoc.reservas.items()
                        if i < self.id and pos is not None]

        def factible(x):
            return (x in dist_aqui and x in dist_base
                    and dist_aqui[x] + dist_base[x] <= self.bateria - self.p.margen_bateria)

        # Compromiso: si el objetivo actual sigue siendo válido, el dron lo
        # mantiene en lugar de replanear en cada paso. Solo lo cede si un dron
        # de mayor prioridad (identificador menor) reservó algo demasiado cerca.
        # Sin esta regla, drones idénticos que se enteran con retraso de las
        # reservas ajenas cambian de objetivo todos a la vez y oscilan.
        r = self.reserva
        if r is not None and r != self.posicion and factible(r):
            sigue_valida = (conoc.mapa.get(r) == Observacion.SOSPECHA
                            or (self._transitable(conoc, r) and self._es_frontera(conoc, r)))
            en_conflicto = any(abs(r[0] - q[0]) + abs(r[1] - q[1]) <= (0 if conoc.mapa.get(r) == Observacion.SOSPECHA
                                                                          else self.p.radio_reserva)
                               for q in prioritarias)
            if sigue_valida and not en_conflicto:
                return self._primer_paso(conoc, r), r, f"Mantiene su objetivo {r}."

        sospechas = [x for x, o in conoc.mapa.items() if o == Observacion.SOSPECHA and x not in ajenas]
        fronteras = [x for x in conoc.mapa if self._transitable(conoc, x) and self._es_frontera(conoc, x)]
        lejos = [x for x in fronteras
                 if all(abs(x[0] - r[0]) + abs(x[1] - r[1]) > self.p.radio_reserva for r in ajenas)]
        libres = [x for x in fronteras if x not in ajenas]

        for tipo, candidatos in (("sospecha", sospechas), ("frontera", lejos), ("frontera", libres)):
            factibles = [x for x in candidatos if factible(x)]
            if factibles:
                objetivo = min(factibles, key=lambda x: (dist_aqui[x], x))
                if tipo == "sospecha":
                    motivo = f"Reserva la sospecha en {objetivo} ({dist_aqui[objetivo]} pasos) para confirmarla."
                else:
                    motivo = f"Reserva la frontera {objetivo}, lejos de las reservas de sus compañeros."
                return self._primer_paso(conoc, objetivo), objetivo, motivo

        pendientes = bool(sospechas or libres)
        if self.posicion != BASE:
            self.regresando = True
            return self._primer_paso(conoc, BASE), None, "Nada alcanzable con su batería; regresa a la base."
        if pendientes and self.bateria < self.p.bateria_maxima:
            self.regresando = True
            return "esperar", None, "En la base con batería parcial; se queda a recargar."
        return "aterrizar", None, "No quedan sospechas ni fronteras libres: aterriza."

    # --- actuar -----------------------------------------------------------
    def actuar(self, accion: str, cultivo: Cultivo, paso: int, sobrevoladas_equipo: set) -> bool:
        if accion == "aterrizar":
            self.estado = "terminado"
            return False
        if accion == "esperar" and self.posicion == BASE and self.regresando:
            self._iniciar_recarga()
            return False
        destino = destino_de(self.posicion, accion)
        if accion != "esperar" and cultivo.es_transitable(destino):
            if destino in sobrevoladas_equipo:
                self.desempeno.revisitas += 1
            self.posicion = destino
            sobrevoladas_equipo.add(destino)
        self.bateria -= 1
        self.desempeno.pasos_vuelo += 1
        confirmo = cultivo.sobrevolar(self.posicion, paso)
        if confirmo:
            self.desempeno.focos_confirmados += 1
        if self.posicion == BASE and self.regresando:
            self._iniciar_recarga()
        elif self.bateria <= 0 and self.posicion != BASE:
            self.estado = "perdido"
            self.desempeno.perdido = True
        return confirmo

    def _iniciar_recarga(self) -> None:
        self.regresando = False
        self.estado = "recargando"
        self.recarga_restante = self.p.tiempo_recarga
        self.desempeno.regresos_a_base += 1

    def paso_recarga(self) -> None:
        self.recarga_restante -= 1
        if self.recarga_restante <= 0:
            self.bateria = self.p.bateria_maxima
            self.estado = "volando"


def observacion_propia(cultivo: Cultivo, pos) -> dict:
    """Lo que el dron sabe con certeza de la parcela sobre la que está."""
    return {pos: cultivo._observar(pos)}


# =============================================================================
# SECCIÓN 4. Mecanismo A: pizarrón compartido en la estación base
# =============================================================================

def _nuevo_episodio(p: ParametrosPractica):
    cultivo = Cultivo(p.tamano_cultivo, p.densidad_restringidas, p.num_focos_iniciales,
                      p.prob_propagacion, p.semilla)
    drones = [Dron(i, p, random.Random(p.semilla * 100 + i)) for i in range(p.num_drones)]
    return cultivo, drones


def ejecutar_pizarron(p: ParametrosPractica) -> dict:
    """
    Los drones se procesan uno por uno dentro de cada paso y el pizarrón
    se actualiza ANTES de que decida el siguiente. Por eso nunca hay dos
    reservas iguales: no puede haber disputas.
    """
    cultivo, drones = _nuevo_episodio(p)
    restringidas = cultivo.restringidas()
    pizarron = Conocimiento()
    sobrevoladas_equipo = {BASE}
    trayectorias = {d.id: [BASE] for d in drones}
    bitacora = []

    for paso in range(1, p.pasos_mision + 1):
        for nuevo in cultivo.propagar(paso):
            bitacora.append({"paso": paso, "id": None, "tipo": "propagacion",
                             "texto": f"La plaga se propaga a {nuevo}."})
        for d in drones:
            if d.estado == "recargando":
                d.paso_recarga()
            elif d.estado == "volando":
                percepto, _ = d.percibir(cultivo, pizarron)
                accion, reserva, motivo = d.decidir(percepto, pizarron)
                confirmo = d.actuar(accion, cultivo, paso, sobrevoladas_equipo)
                # Escritura instantánea en el pizarrón
                pizarron.incorporar(observacion_propia(cultivo, d.posicion))
                pizarron.sobrevoladas.add(d.posicion)
                d.reserva = reserva if d.estado == "volando" else None
                pizarron.reservas[d.id] = (d.reserva, paso)
                bitacora.append({"paso": paso, "id": d.id, "tipo": "decision", "texto": motivo})
                if confirmo:
                    bitacora.append({"paso": paso, "id": d.id, "tipo": "confirmacion",
                                     "texto": f"Dron {d.id} confirma un foco en {d.posicion}."})
            trayectorias[d.id].append(d.posicion)

    return _empaquetar(cultivo, drones, trayectorias, restringidas, bitacora, p,
                       mensajes=[], disputas=[], en_transito={})


# =============================================================================
# SECCIÓN 5. Mecanismo B: mensajería directa
# =============================================================================

def _aplicar_mensaje(m: Mensaje, conocs: dict) -> None:
    destinatarios = [m.receptor] if m.receptor is not None else [i for i in conocs if i != m.emisor]
    for i in destinatarios:
        k = conocs[i]
        if m.performativo == "informar":
            k.incorporar(m.contenido["celdas"])
            k.sobrevoladas.add(m.contenido["posicion"])
            k.reservas[m.emisor] = (m.contenido["reserva"], m.paso_envio)
        elif m.performativo == "proponer":
            # El perdedor reconoce la reserva del ganador y elegirá otro objetivo.
            k.reservas[m.emisor] = (m.contenido["posicion"], m.paso_envio)


def ejecutar_mensajeria(p: ParametrosPractica) -> dict:
    """
    Dos fases por paso, como en la Práctica 2 original: (1) todos perciben
    y deciden con su conocimiento privado; (2) se ejecutan los movimientos
    y se envían los avisos, que llegan retraso_mensaje pasos después.
    """
    cultivo, drones = _nuevo_episodio(p)
    restringidas = cultivo.restringidas()
    conocs = {d.id: Conocimiento() for d in drones}
    sobrevoladas_equipo = {BASE}
    trayectorias = {d.id: [BASE] for d in drones}
    bitacora, historial, pendientes, disputas = [], [], [], []
    en_transito = {}

    def enviar(m: Mensaje) -> None:
        pendientes.append(m)
        historial.append(m)

    for paso in range(1, p.pasos_mision + 1):
        for nuevo in cultivo.propagar(paso):
            bitacora.append({"paso": paso, "id": None, "tipo": "propagacion",
                             "texto": f"La plaga se propaga a {nuevo}."})

        listos = [m for m in pendientes if m.paso_entrega <= paso]
        pendientes[:] = [m for m in pendientes if m.paso_entrega > paso]
        for m in listos:
            _aplicar_mensaje(m, conocs)

        # Fase 1: percepción y decisión de todos.
        decisiones, nuevas_reservas = [], {}
        for d in drones:
            if d.estado == "recargando":
                d.paso_recarga()
            elif d.estado == "volando":
                k = conocs[d.id]
                percepto, cambios = d.percibir(cultivo, k)
                accion, reserva, motivo = d.decidir(percepto, k)
                if reserva is not None and reserva != d.reserva:
                    nuevas_reservas.setdefault(reserva, []).append(d.id)
                decisiones.append((d, accion, reserva, motivo, cambios))

        # Fase 2: ejecución y avisos por radio.
        for d, accion, reserva, motivo, cambios in decisiones:
            k = conocs[d.id]
            reserva_anterior = d.reserva
            confirmo = d.actuar(accion, cultivo, paso, sobrevoladas_equipo)
            propia = observacion_propia(cultivo, d.posicion)
            cambios.update({x: o for x, o in propia.items() if k.mapa.get(x) != o})
            k.incorporar(propia)
            k.sobrevoladas.add(d.posicion)
            d.reserva = reserva if d.estado == "volando" else None
            k.reservas[d.id] = (d.reserva, paso)
            bitacora.append({"paso": paso, "id": d.id, "tipo": "decision", "texto": motivo})
            if confirmo:
                bitacora.append({"paso": paso, "id": d.id, "tipo": "confirmacion",
                                 "texto": f"Dron {d.id} confirma un foco en {d.posicion}."})
            if d.estado != "perdido" and (cambios or d.reserva != reserva_anterior):
                enviar(Mensaje(d.id, None, "informar",
                               {"celdas": cambios, "posicion": d.posicion, "reserva": d.reserva},
                               paso, paso + p.retraso_mensaje))

        # Disputas: dos o más drones reservaron lo mismo sin saberlo.
        for pos, ids in nuevas_reservas.items():
            if len(ids) > 1:
                ganador = min(ids)
                for perdedor in ids:
                    if perdedor == ganador:
                        continue
                    enviar(Mensaje(ganador, perdedor, "proponer", {"posicion": pos},
                                   paso, paso + p.retraso_mensaje))
                    disputas.append({"paso": paso, "posicion": pos, "ganador": ganador, "perdedor": perdedor})
                    bitacora.append({"paso": paso, "id": perdedor, "tipo": "disputa",
                                     "texto": f"Disputa: drones {ganador} y {perdedor} reservaron {pos} en el "
                                              f"mismo paso. El dron {ganador} conserva la reserva."})

        en_transito[paso] = len(pendientes)
        for d in drones:
            trayectorias[d.id].append(d.posicion)

    return _empaquetar(cultivo, drones, trayectorias, restringidas, bitacora, p,
                       mensajes=historial, disputas=disputas, en_transito=en_transito)


# =============================================================================
# SECCIÓN 6. Resultados y medida de desempeño del equipo
# =============================================================================

def _empaquetar(cultivo, drones, trayectorias, restringidas, bitacora, p, mensajes, disputas, en_transito) -> dict:
    cultivables = cultivo.parcelas_cultivables()
    inspeccionadas = sum(1 for x in cultivo.inspeccionadas if cultivo.estado(x) != Estado.RESTRINGIDA)
    confirmados = len(cultivo.confirmacion_foco)
    tiempos = [cultivo.confirmacion_foco[x] - cultivo.aparicion_foco[x] for x in cultivo.confirmacion_foco]
    infestadas = cultivo.infestadas_sin_detectar()
    energia = sum(d.desempeno.pasos_vuelo for d in drones)
    redundancia = sum(d.desempeno.revisitas for d in drones)
    perdidos = sum(1 for d in drones if d.desempeno.perdido)
    puntaje = (confirmados * p.peso_foco_confirmado
               + inspeccionadas * p.peso_parcela_inspeccionada
               - infestadas * p.peso_parcela_infestada
               - energia * p.peso_energia
               - perdidos * p.peso_dron_perdido
               - redundancia * p.peso_redundancia)
    return {
        "resumen": {
            "cobertura_pct": 100.0 * inspeccionadas / cultivables,
            "focos_totales": len(cultivo.aparicion_foco),
            "focos_confirmados": confirmados,
            "infestadas_sin_detectar": infestadas,
            "tiempo_deteccion": statistics.mean(tiempos) if tiempos else float("nan"),
            "energia": energia,
            "redundancia": redundancia,
            "mensajes": len(mensajes),
            "disputas": len(disputas),
            "drones_perdidos": perdidos,
            "puntaje": puntaje,
        },
        "trayectorias": trayectorias,
        "restringidas": restringidas,
        "aparicion_foco": dict(cultivo.aparicion_foco),
        "confirmacion_foco": dict(cultivo.confirmacion_foco),
        "bitacora": bitacora,
        "mensajes": mensajes,
        "disputas": disputas,
        "en_transito": en_transito,
    }


MECANISMOS = [("Pizarrón compartido", ejecutar_pizarron), ("Mensajería directa", ejecutar_mensajeria)]


def comparar_mecanismos(p: ParametrosPractica) -> dict:
    return {nombre: fn(p) for nombre, fn in MECANISMOS}


def experimento_multisemilla(p: ParametrosPractica) -> dict:
    """Cruza 2 arquitecturas x 2 mecanismos sobre las mismas semillas."""
    datos = {}
    for arq in ("reactiva", "basado_en_modelos"):
        for nombre, fn in MECANISMOS:
            datos[(arq, nombre)] = [fn(replace(p, semilla=s, arquitectura=arq))["resumen"]
                                    for s in range(1, p.num_semillas_experimento + 1)]
    return datos


# =============================================================================
# SECCIÓN 7. Reporte en terminal
# =============================================================================

METRICAS = [
    ("cobertura_pct", "Cultivo inspeccionado (%)"),
    ("focos_totales", "Focos totales"),
    ("focos_confirmados", "Focos confirmados"),
    ("infestadas_sin_detectar", "Infestadas sin detectar"),
    ("tiempo_deteccion", "Tiempo medio a detección"),
    ("energia", "Pasos de vuelo (energía)"),
    ("redundancia", "Redundancia (sobrevuelos repetidos)"),
    ("mensajes", "Mensajes enviados"),
    ("disputas", "Disputas resueltas"),
    ("drones_perdidos", "Drones perdidos"),
    ("puntaje", "Puntaje de desempeño"),
]


def _fmt(x) -> str:
    if isinstance(x, float):
        return "n/d" if x != x else f"{x:.1f}"
    return str(x)


def imprimir_comparacion(res: dict, p: ParametrosPractica) -> None:
    print("=" * 80)
    print("AgroSwarm - PRÁCTICA 2: COORDINACIÓN DE LA FLOTA DE DRONES")
    print(f"Cultivo {p.tamano_cultivo}x{p.tamano_cultivo} | Drones {p.num_drones} | Arquitectura: {p.arquitectura} | "
          f"Retraso de mensaje {p.retraso_mensaje} | Semilla {p.semilla}")
    print("=" * 80)
    a, b = list(res)
    print(f"{'Métrica':<40}{a:>20}{b:>20}")
    print("-" * 80)
    for clave, etiqueta in METRICAS:
        print(f"{etiqueta:<40}{_fmt(res[a]['resumen'][clave]):>20}{_fmt(res[b]['resumen'][clave]):>20}")
    print("=" * 80)


def imprimir_bitacora(res: dict, max_decisiones: int = 3) -> None:
    print("\nBITÁCORA (eventos relevantes)")
    print("-" * 80)
    for nombre, datos in res.items():
        print(f"\n[{nombre}]")
        for e in [e for e in datos["bitacora"] if e["tipo"] == "decision"][:max_decisiones]:
            print(f"  Paso {e['paso']:>3} (Dron {e['id']}): {e['texto']}")
        relevantes = [e for e in datos["bitacora"] if e["tipo"] in ("disputa", "confirmacion")][:8]
        if relevantes:
            print("  (...)")
            for e in relevantes:
                print(f"  Paso {e['paso']:>3}: {e['texto']}")
        if not any(e["tipo"] == "disputa" for e in datos["bitacora"]):
            print("  No hubo disputas con este mecanismo.")


def imprimir_ejemplos_mensajes(res: dict, max_ejemplos: int = 4) -> None:
    datos = res["Mensajería directa"]
    print("\nEJEMPLOS DE MENSAJES (mensajería directa)")
    print("-" * 80)
    for m in datos["mensajes"][:max_ejemplos]:
        destino = f"Dron {m.receptor}" if m.receptor is not None else "toda la flota"
        if m.performativo == "informar":
            resumen = f"{len(m.contenido['celdas'])} observación(es), posición {m.contenido['posicion']}, reserva {m.contenido['reserva']}"
        else:
            resumen = f"ceder la reserva de {m.contenido['posicion']}"
        print(f"  Paso {m.paso_envio:>3} -> {m.paso_entrega:>3}: Dron {m.emisor} '{m.performativo}' a {destino}: {resumen}")
    valores = list(datos["en_transito"].values())
    if valores:
        print(f"  Mensajes en tránsito por paso: promedio {statistics.mean(valores):.1f}, máximo {max(valores)}.")


def imprimir_experimento(datos: dict, p: ParametrosPractica) -> None:
    print("\n" + "=" * 100)
    print(f"EXPERIMENTO: ARQUITECTURA x MECANISMO, {p.num_semillas_experimento} SEMILLAS (promedio ± desv. estándar)")
    print("=" * 100)
    claves = list(datos)
    encabezados = [("Reactiva" if a == "reactiva" else "Modelos") + " + " + ("Pizarrón" if "Piz" in m else "Mensajes")
                   for a, m in claves]
    print(f"{'Métrica':<28}" + "".join(f"{h:>19}" for h in encabezados))
    print("-" * 100)
    for clave, etiqueta in METRICAS:
        fila = f"{etiqueta[:27]:<28}"
        for k in claves:
            vals = [r[clave] for r in datos[k] if r[clave] == r[clave]]
            celda = f"{statistics.mean(vals):.1f} ± {statistics.pstdev(vals):.1f}" if vals else "n/d"
            fila += f"{celda:>19}"
        print(fila)
    print("=" * 100)


# =============================================================================
# SECCIÓN 8. Figura y animación
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
    eje.set_facecolor("#ECFCCB")


def generar_figura(res: dict, p: ParametrosPractica) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("\nNota: matplotlib no está disponible; no se generó la figura.")
        return
    colores = [plt.cm.tab10(i % 10) for i in range(p.num_drones)]
    fig, ejes = plt.subplots(1, 2, figsize=(13, 6.8))
    for eje, (nombre, datos) in zip(ejes, res.items()):
        _preparar_eje(eje, p.tamano_cultivo, nombre)
        if datos["restringidas"]:
            eje.scatter([x[1] for x in datos["restringidas"]], [x[0] for x in datos["restringidas"]],
                        marker="s", s=420, color="#4B5563", zorder=2)
        for i, tr in datos["trayectorias"].items():
            desfase = (i - (p.num_drones - 1) / 2) * 0.12   # separa visualmente recorridos superpuestos
            eje.plot([x[1] + desfase for x in tr], [x[0] + desfase for x in tr], "-",
                     color=colores[i], alpha=0.6, linewidth=1.2, zorder=3)
        for pos, paso in datos["aparicion_foco"].items():
            conf = pos in datos["confirmacion_foco"]
            eje.scatter(pos[1], pos[0], marker="*" if paso == 0 else "D", s=330 if paso == 0 else 120,
                        color="#F59E0B" if conf else "#DC2626", edgecolors="#7F1D1D", zorder=5)
        for d in datos["disputas"]:
            eje.scatter(d["posicion"][1], d["posicion"][0], s=650, facecolors="none",
                        edgecolors="#DC2626", linewidths=2, linestyle="--", zorder=7)
        eje.scatter(0, 0, marker="s", s=300, color="#16A34A", edgecolors="white", zorder=6)
        r = datos["resumen"]
        eje.text(0.5, 1.01,
                 f"Inspeccionado {r['cobertura_pct']:.0f}%  Focos {r['focos_confirmados']}/{r['focos_totales']}  "
                 f"Sin detectar {r['infestadas_sin_detectar']}\nVuelo {r['energia']}  Redundancia {r['redundancia']}  "
                 f"Mensajes {r['mensajes']}  Disputas {r['disputas']}  Puntaje {r['puntaje']:.1f}",
                 transform=eje.transAxes, ha="center", va="bottom", fontsize=8.5, family="monospace")
    leyenda = [Patch(facecolor="#4B5563", label="Zona no sobrevolable"),
               Patch(facecolor="#16A34A", label="Estación base"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="#F59E0B", markersize=14, label="Foco confirmado"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="#DC2626", markersize=14, label="Foco sin detectar"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#DC2626", markersize=12, label="Disputa")]
    leyenda += [Line2D([0], [0], color=colores[i], label=f"Dron {i}") for i in range(p.num_drones)]
    fig.legend(handles=leyenda, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(f"AgroSwarm - Práctica 2: pizarrón compartido frente a mensajería (arquitectura {p.arquitectura.replace('_', ' ')})", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    ruta = _ruta_junto_al_script(p.ruta_figura)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura generada: {ruta}")


def generar_animacion(res: dict, p: ParametrosPractica, fps: int = 10) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import PIL  # noqa: F401
    except ImportError:
        print("\nNota: falta matplotlib o Pillow (pip install pillow); no se generó la animación.")
        return
    colores = [plt.cm.tab10(i % 10) for i in range(p.num_drones)]
    fig, ejes = plt.subplots(1, 2, figsize=(13, 6.8))
    refs = {}
    for eje, (nombre, datos) in zip(ejes, res.items()):
        _preparar_eje(eje, p.tamano_cultivo, nombre)
        if datos["restringidas"]:
            eje.scatter([x[1] for x in datos["restringidas"]], [x[0] for x in datos["restringidas"]],
                        marker="s", s=420, color="#4B5563", zorder=2)
        eje.scatter(0, 0, marker="s", s=300, color="#16A34A", zorder=2)
        focos = list(datos["aparicion_foco"])
        sc = eje.scatter([x[1] for x in focos], [x[0] for x in focos], marker="*", s=300, zorder=5)
        lineas = {i: eje.plot([], [], "-", color=colores[i], alpha=0.5)[0] for i in datos["trayectorias"]}
        puntos = {i: eje.scatter([], [], s=170, color=colores[i], edgecolors="white", zorder=6) for i in datos["trayectorias"]}
        txt = eje.text(0.5, 1.01, "", transform=eje.transAxes, ha="center", va="bottom", fontsize=9, family="monospace")
        refs[nombre] = (sc, focos, lineas, puntos, txt)
    fig.suptitle("AgroSwarm - Práctica 2: flota de drones paso a paso", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def actualizar(t):
        artistas = []
        for nombre, datos in res.items():
            sc, focos, lineas, puntos, txt = refs[nombre]
            for i, tr in datos["trayectorias"].items():
                tramo = tr[: t + 1]
                lineas[i].set_data([x[1] for x in tramo], [x[0] for x in tramo])
                puntos[i].set_offsets([[tramo[-1][1], tramo[-1][0]]])
            cols = []
            for f in focos:
                if datos["aparicion_foco"][f] > t:
                    cols.append((0, 0, 0, 0))
                elif datos["confirmacion_foco"].get(f, 10**9) <= t:
                    cols.append("#F59E0B")
                else:
                    cols.append("#DC2626")
            sc.set_facecolor(cols)
            sc.set_edgecolor(["#7F1D1D" if c != (0, 0, 0, 0) else (0, 0, 0, 0) for c in cols])
            txt.set_text(f"Paso {t}/{p.pasos_mision}  Focos activos {sum(c == '#DC2626' for c in cols)}  "
                         f"Mensajes en tránsito {datos['en_transito'].get(t, 0)}")
            artistas += list(lineas.values()) + list(puntos.values()) + [sc, txt]
        return artistas

    anim = animation.FuncAnimation(fig, actualizar, frames=p.pasos_mision + 1, interval=1000 / fps)
    ruta = _ruta_junto_al_script(p.ruta_animacion)
    try:
        anim.save(ruta, writer=animation.PillowWriter(fps=fps))
        print(f"Animación generada: {ruta}")
    finally:
        plt.close(fig)


# =============================================================================
# SECCIÓN 9. Punto de entrada
# =============================================================================

if __name__ == "__main__":
    parametros = ParametrosPractica()
    resultados = comparar_mecanismos(parametros)
    imprimir_comparacion(resultados, parametros)
    imprimir_bitacora(resultados)
    imprimir_ejemplos_mensajes(resultados)
    imprimir_experimento(experimento_multisemilla(parametros), parametros)
    generar_figura(resultados, parametros)
    generar_animacion(resultados, parametros)
