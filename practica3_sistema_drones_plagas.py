"""
================================================================================
Laboratorio de Sistemas Multiagente - Portafolio (PIA)
Proyecto: AgroSwarm - Sistema multiagente de drones para la detección
          temprana de plagas en cultivos
Práctica 3 adaptada - Fase 3: Sistemas multiagente aplicados
Sistema integrado: arquitectura + coordinación + robustez
================================================================================

PROPÓSITO DEL PROGRAMA

Este programa integra las dos fases anteriores en el sistema completo:

    - Arquitectura de cada dron: basada en modelos (elegida con la
      evidencia de la Práctica 1), con regreso planeado a la base.
    - Coordinación de la flota: pizarrón compartido en la estación base
      o mensajería directa (comparados en la Práctica 2), con reservas de
      objetivos, compromiso y prioridad por identificador.

y somete al sistema a cinco condiciones de robustez propias del trabajo
de drones en campo:

    1. Falla de dron (prob_falla_dron): falla mecánica o aterrizaje de
       emergencia. El dron queda fuera de servicio donde esté.

    2. Pérdida de enlace (prob_perdida_mensaje): un aviso por radio no
       llega nunca. A diferencia del script original, aquí la pérdida
       afecta A AMBOS mecanismos: el pizarrón vive en la estación base y
       los drones le escriben por radio, así que una escritura también
       puede perderse.

    3. Zonas restringidas nuevas (prob_zona_dinamica): ráfagas de viento,
       maquinaria trabajando o una fumigación en curso vuelven no
       sobrevolable una parcela que antes lo era.

    4. Sensor con error (prob_falso_negativo, prob_falso_positivo): la
       cámara puede no ver síntomas que sí existen o ver síntomas donde no
       los hay. El sobrevuelo directo de la parcela siempre es exacto.

    5. La plaga sigue propagándose mientras el sistema opera (igual que en
       las Prácticas 1 y 2).

MANEJO DE FALLAS (parámetro ttl_reserva)

Si un dron falla con una parcela reservada, nadie más iría a ella y ese
foco quedaría sin atender. Para evitarlo, cada reserva caduca si su dueño
no da señales de vida durante ttl_reserva pasos:
    - Con pizarrón, la estación base borra las reservas que no se
      refrescan (cada escritura de un dron funciona como latido).
    - Con mensajería, cada dron envía un aviso al menos cada ttl_reserva/2
      pasos aunque no tenga nada nuevo, y sus compañeros ignoran las
      reservas de quien lleva más de ttl_reserva pasos sin reportarse.
Con ttl_reserva = None se desactiva este manejo, para medir su efecto.

ARCHIVOS QUE GENERA

    1. Reporte en terminal: tabla comparativa, bitácora de eventos de
       robustez y experimento por escenarios con varias semillas.
    2. comparacion_practica3.png y simulacion_practica3.gif.

FORMA DE EJECUCIÓN

    python practica3_sistema_drones_plagas.py
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
    # --- Cultivo (igual que en las Prácticas 1 y 2) ---
    tamano_cultivo: int = 12
    densidad_restringidas: float = 0.12
    num_focos_iniciales: int = 4
    prob_propagacion: float = 0.02
    semilla: int = 42

    # --- Drones ---
    num_drones: int = 3
    bateria_maxima: int = 40
    tiempo_recarga: int = 5
    margen_bateria: int = 2
    pasos_mision: int = 150

    # --- Coordinación ---
    retraso_mensaje: int = 1
    radio_reserva: int = 2

    # --- Robustez (nuevo en esta práctica) ---
    prob_falla_dron: float = 0.004       # por dron y por paso de vuelo
    prob_perdida_mensaje: float = 0.2    # por mensaje o escritura al pizarrón
    prob_zona_dinamica: float = 0.02     # por paso: aparece una zona no sobrevolable
    prob_falso_negativo: float = 0.15    # la cámara no ve una plaga que sí está
    prob_falso_positivo: float = 0.03    # la cámara ve síntomas en una parcela sana
    ttl_reserva: Optional[int] = 6       # None desactiva la caducidad de reservas

    # --- Medida de desempeño del EQUIPO ---
    peso_foco_confirmado: float = 10.0
    peso_parcela_inspeccionada: float = 1.0
    peso_parcela_infestada: float = 5.0
    peso_energia: float = 0.2
    peso_dron_perdido: float = 20.0      # dron perdido por batería agotada (error de decisión)
    peso_dron_fallido: float = 20.0      # dron fuera de servicio por falla (pérdida de equipo para la flota)
    peso_redundancia: float = 0.5

    num_semillas_experimento: int = 30
    ruta_figura: str = "comparacion_practica3.png"
    ruta_animacion: str = "simulacion_practica3.gif"


def sin_robustez(p: ParametrosPractica) -> ParametrosPractica:
    return replace(p, prob_falla_dron=0.0, prob_perdida_mensaje=0.0, prob_zona_dinamica=0.0,
                   prob_falso_negativo=0.0, prob_falso_positivo=0.0)


# =============================================================================
# SECCIÓN 1. El cultivo, ahora con zonas dinámicas y sensor con error
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
    prob_falso_negativo: float = 0.0
    prob_falso_positivo: float = 0.0

    terreno: list = field(init=False)
    aparicion_foco: dict = field(init=False)
    confirmacion_foco: dict = field(init=False)
    inspeccionadas: set = field(init=False)
    restringidas_iniciales: list = field(init=False)
    zonas_dinamicas: dict = field(init=False)   # posición -> paso en que se volvió no sobrevolable

    def __post_init__(self) -> None:
        rng = random.Random(self.semilla)
        self.rng_dinamica = random.Random(self.semilla + 1000)   # propagación de la plaga
        self.rng_sensor = random.Random(self.semilla + 3000)     # errores de la cámara
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
        self.restringidas_iniciales = self.restringidas()
        self.zonas_dinamicas = {}

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
        """Ventana de 3x3. Las parcelas vecinas se ven desde lejos y pueden
        reportarse con error; la parcela de abajo se ve sin error."""
        f, c = pos
        percepto = {}
        for df in (-1, 0, 1):
            for dc in (-1, 0, 1):
                v = (f + df, c + dc)
                if self.dentro(v):
                    self.inspeccionadas.add(v)
                    obs = self.observar_exacto(v)
                    if v != pos:
                        if obs == Observacion.SOSPECHA and self.rng_sensor.random() < self.prob_falso_negativo:
                            obs = Observacion.SANA
                        elif obs == Observacion.SANA and self.rng_sensor.random() < self.prob_falso_positivo:
                            obs = Observacion.SOSPECHA
                    percepto[v] = obs
        return percepto

    def observar_exacto(self, pos) -> Observacion:
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

    def agregar_zona_dinamica(self, rng: random.Random, ocupadas: set, paso: int):
        candidatas = [(f, c) for f in range(self.tamano) for c in range(self.tamano)
                      if self.terreno[f][c] == Estado.SANA and (f, c) != BASE
                      and abs(f) + abs(c) > 1 and (f, c) not in ocupadas]
        if not candidatas:
            return None
        pos = rng.choice(candidatas)
        self.terreno[pos[0]][pos[1]] = Estado.RESTRINGIDA
        self.zonas_dinamicas[pos] = paso
        return pos

    def parcelas_cultivables(self) -> int:
        return sum(1 for f in range(self.tamano) for c in range(self.tamano)
                   if self.terreno[f][c] != Estado.RESTRINGIDA)

    def infestadas_sin_detectar(self) -> int:
        return sum(1 for p in self.aparicion_foco if self.estado(p) == Estado.PLAGA)


# =============================================================================
# SECCIÓN 2. Conocimiento, mensajes y desempeño
# =============================================================================

@dataclass
class Conocimiento:
    mapa: dict = field(default_factory=lambda: {BASE: Observacion.SANA})
    sobrevoladas: set = field(default_factory=lambda: {BASE})
    reservas: dict = field(default_factory=dict)   # id_dron -> (parcela o None, paso del último reporte)

    def incorporar(self, celdas: dict) -> None:
        for pos, obs in celdas.items():
            if self.mapa.get(pos) != Observacion.CONFIRMADA:
                self.mapa[pos] = obs

    def copia(self) -> "Conocimiento":
        return Conocimiento(dict(self.mapa), set(self.sobrevoladas), dict(self.reservas))


@dataclass
class Mensaje:
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
    perdido: bool = False      # sin batería fuera de la base
    fallido: bool = False      # falla mecánica (condición de robustez)
    falsos_positivos: int = 0  # sospechas que resultaron sanas al sobrevolarlas


# =============================================================================
# SECCIÓN 3. El dron basado en modelos (Prácticas 1 y 2)
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
    def __init__(self, id_dron: int, p: ParametrosPractica):
        self.id = id_dron
        self.p = p
        self.posicion = BASE
        self.bateria = p.bateria_maxima
        self.estado = "volando"   # volando | recargando | terminado | perdido | fallido
        self.recarga_restante = 0
        self.regresando = False
        self.reserva = None
        self.ultimo_envio = 0
        self.desempeno = Desempeno()

    def percibir(self, cultivo: Cultivo, conoc: Conocimiento) -> tuple[dict, dict]:
        percepto = cultivo.percibir(self.posicion)
        cambios = {x: o for x, o in percepto.items() if conoc.mapa.get(x) != o}
        conoc.incorporar(percepto)
        return percepto, cambios

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

    def _reservas_vigentes(self, conoc, paso) -> dict:
        """Reservas ajenas que siguen vigentes (manejo de fallas por caducidad)."""
        ttl = self.p.ttl_reserva
        return {i: pos for i, (pos, visto) in conoc.reservas.items()
                if i != self.id and pos is not None and (ttl is None or paso - visto <= ttl)}

    def decidir(self, conoc: Conocimiento, paso: int) -> tuple[str, Optional[tuple], str]:
        dist_aqui = self._bfs(conoc, self.posicion)
        dist_base = self._bfs(conoc, BASE)

        # Si una zona nueva cortó la ruta conocida a la base, el dron no
        # la encuentra en su mapa: regresa por la ruta más corta posible
        # en cuanto la descubra (aquí, usa la distancia en línea recta como
        # estimación conservadora).
        regreso = dist_base.get(self.posicion, self.posicion[0] + self.posicion[1])

        if self.posicion != BASE and self.bateria <= regreso + self.p.margen_bateria:
            self.regresando = True
            return self._primer_paso(conoc, BASE), None, \
                f"Batería {self.bateria}: la ruta conocida a la base mide {regreso}; regresa."

        vigentes = self._reservas_vigentes(conoc, paso)
        ajenas = list(vigentes.values())
        prioritarias = [pos for i, pos in vigentes.items() if i < self.id]

        def factible(x):
            return (x in dist_aqui and x in dist_base
                    and dist_aqui[x] + dist_base[x] <= self.bateria - self.p.margen_bateria)

        r = self.reserva
        if r is not None and r != self.posicion and factible(r):
            es_sospecha = conoc.mapa.get(r) == Observacion.SOSPECHA
            sigue_valida = es_sospecha or (self._transitable(conoc, r) and self._es_frontera(conoc, r))
            radio = 0 if es_sospecha else self.p.radio_reserva
            en_conflicto = any(abs(r[0] - q[0]) + abs(r[1] - q[1]) <= radio for q in prioritarias)
            if sigue_valida and not en_conflicto:
                return self._primer_paso(conoc, r), r, f"Mantiene su objetivo {r}."

        sospechas = [x for x, o in conoc.mapa.items() if o == Observacion.SOSPECHA and x not in ajenas]
        fronteras = [x for x in conoc.mapa if self._transitable(conoc, x) and self._es_frontera(conoc, x)]
        lejos = [x for x in fronteras
                 if all(abs(x[0] - q[0]) + abs(x[1] - q[1]) > self.p.radio_reserva for q in ajenas)]
        libres = [x for x in fronteras if x not in ajenas]

        for tipo, candidatos in (("sospecha", sospechas), ("frontera", lejos), ("frontera", libres)):
            factibles = [x for x in candidatos if factible(x)]
            if factibles:
                objetivo = min(factibles, key=lambda x: (dist_aqui[x], x))
                if tipo == "sospecha":
                    motivo = f"Reserva la sospecha en {objetivo} para confirmarla."
                else:
                    motivo = f"Reserva la frontera {objetivo}, lejos de las reservas vigentes de sus compañeros."
                return self._primer_paso(conoc, objetivo), objetivo, motivo

        pendientes = bool(sospechas or libres)
        if self.posicion != BASE:
            self.regresando = True
            return self._primer_paso(conoc, BASE), None, "Nada alcanzable con su batería; regresa a la base."
        if pendientes and self.bateria < self.p.bateria_maxima:
            self.regresando = True
            return "esperar", None, "En la base con batería parcial; se queda a recargar."
        return "aterrizar", None, "No quedan sospechas ni fronteras libres: aterriza."

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
        # Si el destino se volvió no sobrevolable, el dron no avanza: gasta
        # el paso y lo descubre en su siguiente percepción.
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


# =============================================================================
# SECCIÓN 4. Eventos de robustez comunes a ambos mecanismos
# =============================================================================

def _eventos_del_mundo(cultivo, drones, rng, p, paso, bitacora) -> None:
    for nuevo in cultivo.propagar(paso):
        bitacora.append({"paso": paso, "id": None, "tipo": "propagacion",
                         "texto": f"La plaga se propaga a {nuevo}."})
    if p.prob_zona_dinamica > 0 and rng.random() < p.prob_zona_dinamica:
        pos = cultivo.agregar_zona_dinamica(rng, {d.posicion for d in drones}, paso)
        if pos is not None:
            bitacora.append({"paso": paso, "id": None, "tipo": "zona_dinamica",
                             "texto": f"La parcela {pos} se vuelve no sobrevolable (viento o maquinaria)."})


def _falla(d: Dron, rng, p, paso, bitacora) -> bool:
    if d.estado == "volando" and p.prob_falla_dron > 0 and rng.random() < p.prob_falla_dron:
        d.estado = "fallido"
        d.desempeno.fallido = True
        bitacora.append({"paso": paso, "id": d.id, "tipo": "falla",
                         "texto": f"El dron {d.id} sufre una falla y queda fuera de servicio en {d.posicion}"
                                  + (f" con la reserva {d.reserva} pendiente." if d.reserva else ".")})
        return True
    return False


def _registrar_sobrevuelo(d: Dron, cultivo: Cultivo, conoc: Conocimiento) -> dict:
    """Corrige el mapa con lo que el dron ve directamente debajo."""
    if conoc.mapa.get(d.posicion) == Observacion.SOSPECHA and cultivo.estado(d.posicion) == Estado.SANA:
        d.desempeno.falsos_positivos += 1
    propia = {d.posicion: cultivo.observar_exacto(d.posicion)}
    conoc.incorporar(propia)
    conoc.sobrevoladas.add(d.posicion)
    return propia


def _nuevo_episodio(p: ParametrosPractica):
    cultivo = Cultivo(p.tamano_cultivo, p.densidad_restringidas, p.num_focos_iniciales,
                      p.prob_propagacion, p.semilla, p.prob_falso_negativo, p.prob_falso_positivo)
    drones = [Dron(i, p) for i in range(p.num_drones)]
    return cultivo, drones, random.Random(p.semilla + 2000)


# =============================================================================
# SECCIÓN 5. Mecanismo A: pizarrón compartido en la estación base
# =============================================================================

def ejecutar_pizarron(p: ParametrosPractica) -> dict:
    """
    Cada dron LEE el pizarrón (enlace de bajada de la base, confiable),
    combina esa lectura con lo que ve, decide y actúa, y después ESCRIBE
    en el pizarrón por radio. Esa escritura puede perderse.
    """
    cultivo, drones, rng = _nuevo_episodio(p)
    pizarron = Conocimiento()
    sobrevoladas_equipo = {BASE}
    trayectorias = {d.id: [BASE] for d in drones}
    bitacora = []
    escrituras = perdidos = 0

    for paso in range(1, p.pasos_mision + 1):
        _eventos_del_mundo(cultivo, drones, rng, p, paso, bitacora)
        for d in drones:
            if d.estado == "recargando":
                d.paso_recarga()
            elif d.estado == "volando" and not _falla(d, rng, p, paso, bitacora):
                vista = pizarron.copia()
                _, cambios = d.percibir(cultivo, vista)
                accion, reserva, motivo = d.decidir(vista, paso)
                confirmo = d.actuar(accion, cultivo, paso, sobrevoladas_equipo)
                cambios.update(_registrar_sobrevuelo(d, cultivo, vista))
                d.reserva = reserva if d.estado == "volando" else None
                if d.estado != "perdido":
                    escrituras += 1
                    if rng.random() < p.prob_perdida_mensaje:
                        perdidos += 1
                    else:
                        pizarron.incorporar(cambios)
                        pizarron.sobrevoladas.add(d.posicion)
                        pizarron.reservas[d.id] = (d.reserva, paso)
                bitacora.append({"paso": paso, "id": d.id, "tipo": "decision", "texto": motivo})
                if confirmo:
                    bitacora.append({"paso": paso, "id": d.id, "tipo": "confirmacion",
                                     "texto": f"Dron {d.id} confirma un foco en {d.posicion}."})
            trayectorias[d.id].append(d.posicion)

    return _empaquetar(cultivo, drones, trayectorias, bitacora, p, mensajes=[], enviados=escrituras,
                       perdidos=perdidos, disputas=[], en_transito={})


# =============================================================================
# SECCIÓN 6. Mecanismo B: mensajería directa
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
            k.reservas[m.emisor] = (m.contenido["posicion"], m.paso_envio)


def ejecutar_mensajeria(p: ParametrosPractica) -> dict:
    cultivo, drones, rng = _nuevo_episodio(p)
    conocs = {d.id: Conocimiento() for d in drones}
    sobrevoladas_equipo = {BASE}
    trayectorias = {d.id: [BASE] for d in drones}
    bitacora, historial, pendientes, disputas = [], [], [], []
    en_transito = {}
    contador = {"enviados": 0, "perdidos": 0}
    intervalo_latido = max(1, (p.ttl_reserva or 10**9) // 2)

    def enviar(m: Mensaje) -> None:
        contador["enviados"] += 1
        if rng.random() < p.prob_perdida_mensaje:
            contador["perdidos"] += 1
            return
        pendientes.append(m)
        historial.append(m)

    for paso in range(1, p.pasos_mision + 1):
        _eventos_del_mundo(cultivo, drones, rng, p, paso, bitacora)

        listos = [m for m in pendientes if m.paso_entrega <= paso]
        pendientes[:] = [m for m in pendientes if m.paso_entrega > paso]
        for m in listos:
            _aplicar_mensaje(m, conocs)

        decisiones, nuevas_reservas = [], {}
        for d in drones:
            if d.estado == "recargando":
                d.paso_recarga()
            elif d.estado == "volando" and not _falla(d, rng, p, paso, bitacora):
                k = conocs[d.id]
                _, cambios = d.percibir(cultivo, k)
                accion, reserva, motivo = d.decidir(k, paso)
                if reserva is not None and reserva != d.reserva:
                    nuevas_reservas.setdefault(reserva, []).append(d.id)
                decisiones.append((d, accion, reserva, motivo, cambios))

        for d, accion, reserva, motivo, cambios in decisiones:
            k = conocs[d.id]
            anterior = d.reserva
            confirmo = d.actuar(accion, cultivo, paso, sobrevoladas_equipo)
            cambios.update(_registrar_sobrevuelo(d, cultivo, k))
            d.reserva = reserva if d.estado == "volando" else None
            k.reservas[d.id] = (d.reserva, paso)
            bitacora.append({"paso": paso, "id": d.id, "tipo": "decision", "texto": motivo})
            if confirmo:
                bitacora.append({"paso": paso, "id": d.id, "tipo": "confirmacion",
                                 "texto": f"Dron {d.id} confirma un foco en {d.posicion}."})
            latido = p.ttl_reserva is not None and paso - d.ultimo_envio >= intervalo_latido
            if d.estado != "perdido" and (cambios or d.reserva != anterior or latido):
                d.ultimo_envio = paso
                enviar(Mensaje(d.id, None, "informar",
                               {"celdas": cambios, "posicion": d.posicion, "reserva": d.reserva},
                               paso, paso + p.retraso_mensaje))

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

    return _empaquetar(cultivo, drones, trayectorias, bitacora, p, mensajes=historial,
                       enviados=contador["enviados"], perdidos=contador["perdidos"],
                       disputas=disputas, en_transito=en_transito)


# =============================================================================
# SECCIÓN 7. Resultados y medida de desempeño del equipo
# =============================================================================

def _empaquetar(cultivo, drones, trayectorias, bitacora, p, mensajes, enviados, perdidos,
                disputas, en_transito) -> dict:
    cultivables = cultivo.parcelas_cultivables()
    inspeccionadas = sum(1 for x in cultivo.inspeccionadas if cultivo.estado(x) != Estado.RESTRINGIDA)
    confirmados = len(cultivo.confirmacion_foco)
    tiempos = [cultivo.confirmacion_foco[x] - cultivo.aparicion_foco[x] for x in cultivo.confirmacion_foco]
    infestadas = cultivo.infestadas_sin_detectar()
    energia = sum(d.desempeno.pasos_vuelo for d in drones)
    redundancia = sum(d.desempeno.revisitas for d in drones)
    perdidos_bateria = sum(1 for d in drones if d.desempeno.perdido)
    fallidos = sum(1 for d in drones if d.desempeno.fallido)
    puntaje = (confirmados * p.peso_foco_confirmado
               + inspeccionadas * p.peso_parcela_inspeccionada
               - infestadas * p.peso_parcela_infestada
               - energia * p.peso_energia
               - perdidos_bateria * p.peso_dron_perdido
               - fallidos * p.peso_dron_fallido
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
            "mensajes": enviados,
            "mensajes_perdidos": perdidos,
            "disputas": len(disputas),
            "drones_fallidos": fallidos,
            "drones_perdidos": perdidos_bateria,
            "zonas_nuevas": len(cultivo.zonas_dinamicas),
            "falsos_positivos": sum(d.desempeno.falsos_positivos for d in drones),
            "puntaje": puntaje,
        },
        "trayectorias": trayectorias,
        "restringidas": cultivo.restringidas_iniciales,
        "zonas_dinamicas": dict(cultivo.zonas_dinamicas),
        "aparicion_foco": dict(cultivo.aparicion_foco),
        "confirmacion_foco": dict(cultivo.confirmacion_foco),
        "fallidos": {d.id: d.posicion for d in drones if d.desempeno.fallido},
        "bitacora": bitacora,
        "mensajes": mensajes,
        "disputas": disputas,
        "en_transito": en_transito,
    }


MECANISMOS = [("Pizarrón compartido", ejecutar_pizarron), ("Mensajería directa", ejecutar_mensajeria)]


def comparar_mecanismos(p: ParametrosPractica) -> dict:
    return {nombre: fn(p) for nombre, fn in MECANISMOS}


def escenarios(p: ParametrosPractica) -> list:
    """Cada condición de robustez por separado, todas juntas, y todas juntas sin manejo de fallas."""
    base = sin_robustez(p)
    return [
        ("Sin fallas", base),
        ("Falla de drones", replace(base, prob_falla_dron=p.prob_falla_dron * 2.5)),
        ("Pérdida de enlace", replace(base, prob_perdida_mensaje=p.prob_perdida_mensaje * 1.5)),
        ("Zonas nuevas", replace(base, prob_zona_dinamica=p.prob_zona_dinamica * 1.5)),
        ("Sensor con error", replace(base, prob_falso_negativo=p.prob_falso_negativo,
                                     prob_falso_positivo=p.prob_falso_positivo)),
        ("Todo junto", p),
        ("Todo junto, sin caducidad", replace(p, ttl_reserva=None)),
    ]


def experimento_escenarios(p: ParametrosPractica) -> dict:
    datos = {}
    for nombre_esc, pe in escenarios(p):
        for nombre_mec, fn in MECANISMOS:
            datos[(nombre_esc, nombre_mec)] = [fn(replace(pe, semilla=s))["resumen"]
                                               for s in range(1, p.num_semillas_experimento + 1)]
    return datos


# =============================================================================
# SECCIÓN 8. Reporte en terminal
# =============================================================================

METRICAS = [
    ("cobertura_pct", "Cultivo inspeccionado (%)"),
    ("focos_totales", "Focos totales"),
    ("focos_confirmados", "Focos confirmados"),
    ("infestadas_sin_detectar", "Infestadas sin detectar"),
    ("tiempo_deteccion", "Tiempo medio a detección"),
    ("energia", "Pasos de vuelo (energía)"),
    ("redundancia", "Redundancia"),
    ("mensajes", "Mensajes / escrituras enviadas"),
    ("mensajes_perdidos", "Mensajes / escrituras perdidas"),
    ("disputas", "Disputas resueltas"),
    ("drones_fallidos", "Drones con falla"),
    ("drones_perdidos", "Drones perdidos por batería"),
    ("zonas_nuevas", "Zonas restringidas nuevas"),
    ("falsos_positivos", "Falsos positivos descartados"),
    ("puntaje", "Puntaje de desempeño"),
]


def _fmt(x) -> str:
    if isinstance(x, float):
        return "n/d" if x != x else f"{x:.1f}"
    return str(x)


def imprimir_comparacion(res: dict, p: ParametrosPractica) -> None:
    print("=" * 80)
    print("AgroSwarm - PRÁCTICA 3: SISTEMA INTEGRADO CON ROBUSTEZ")
    print(f"Cultivo {p.tamano_cultivo}x{p.tamano_cultivo} | Drones {p.num_drones} | Arquitectura basada en modelos | Semilla {p.semilla}")
    print(f"Robustez -> falla de dron {p.prob_falla_dron:.1%}/paso | pérdida de enlace {p.prob_perdida_mensaje:.0%} | "
          f"zona nueva {p.prob_zona_dinamica:.0%}/paso")
    print(f"            falso negativo {p.prob_falso_negativo:.0%} | falso positivo {p.prob_falso_positivo:.0%} | "
          f"caducidad de reservas {p.ttl_reserva} pasos")
    print("=" * 80)
    a, b = list(res)
    print(f"{'Métrica':<40}{a:>20}{b:>20}")
    print("-" * 80)
    for clave, etiqueta in METRICAS:
        print(f"{etiqueta:<40}{_fmt(res[a]['resumen'][clave]):>20}{_fmt(res[b]['resumen'][clave]):>20}")
    print("=" * 80)


def imprimir_bitacora(res: dict) -> None:
    print("\nBITÁCORA DE EVENTOS DE ROBUSTEZ")
    print("-" * 80)
    for nombre, datos in res.items():
        print(f"\n[{nombre}]")
        eventos = [e for e in datos["bitacora"] if e["tipo"] in ("falla", "zona_dinamica", "disputa")]
        for e in eventos[:10]:
            print(f"  Paso {e['paso']:>3}: {e['texto']}")
        if len(eventos) > 10:
            print(f"  (...) y {len(eventos) - 10} evento(s) más.")
        if not eventos:
            print("  Sin eventos de robustez.")


def imprimir_experimento(datos: dict, p: ParametrosPractica) -> None:
    print("\n" + "=" * 104)
    print(f"EXPERIMENTO POR ESCENARIO DE ROBUSTEZ, {p.num_semillas_experimento} SEMILLAS (promedios)")
    print("=" * 104)
    print(f"{'':<27}{'---------- Pizarrón compartido ----------':^38}  {'---------- Mensajería directa ----------':^38}")
    sub = f"{'Inspecc.%':>9}{'Sin det.':>9}{'Redund.':>9}{'Puntaje':>11}"
    print(f"{'Escenario':<27}{sub}  {sub}")
    print("-" * 104)
    nombres_esc = []
    for esc, _ in datos:
        if esc not in nombres_esc:
            nombres_esc.append(esc)
    for esc in nombres_esc:
        fila = f"{esc:<27}"
        for mec, _ in MECANISMOS:
            rs = datos[(esc, mec)]
            m = lambda k: statistics.mean(r[k] for r in rs)
            fila += f"{m('cobertura_pct'):>9.1f}{m('infestadas_sin_detectar'):>9.1f}{m('redundancia'):>9.1f}{m('puntaje'):>11.1f}  "
        print(fila)
    print("=" * 104)
    print("Inspecc.% = cultivo inspeccionado; Sin det. = parcelas infestadas sin detectar al final;")
    print("Redund. = sobrevuelos repetidos del equipo.")


# =============================================================================
# SECCIÓN 9. Figura y animación
# =============================================================================

def _preparar_eje(eje, tamano, titulo):
    eje.set_title(titulo, fontsize=13, fontweight="bold", pad=44)
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
    fig, ejes = plt.subplots(1, 2, figsize=(13, 7))
    for eje, (nombre, datos) in zip(ejes, res.items()):
        _preparar_eje(eje, p.tamano_cultivo, nombre)
        if datos["restringidas"]:
            eje.scatter([x[1] for x in datos["restringidas"]], [x[0] for x in datos["restringidas"]],
                        marker="s", s=420, color="#4B5563", zorder=2)
        if datos["zonas_dinamicas"]:
            eje.scatter([x[1] for x in datos["zonas_dinamicas"]], [x[0] for x in datos["zonas_dinamicas"]],
                        marker="s", s=420, color="#92400E", zorder=2)
        for i, tr in datos["trayectorias"].items():
            desfase = (i - (p.num_drones - 1) / 2) * 0.12
            eje.plot([x[1] + desfase for x in tr], [x[0] + desfase for x in tr], "-",
                     color=colores[i], alpha=0.6, linewidth=1.2, zorder=3)
        for i, pos in datos["fallidos"].items():
            eje.scatter(pos[1], pos[0], marker="X", s=260, color=colores[i], edgecolors="black", zorder=8)
        for pos, paso in datos["aparicion_foco"].items():
            conf = pos in datos["confirmacion_foco"]
            eje.scatter(pos[1], pos[0], marker="*" if paso == 0 else "D", s=330 if paso == 0 else 120,
                        color="#F59E0B" if conf else "#DC2626", edgecolors="#7F1D1D", zorder=5)
        eje.scatter(0, 0, marker="s", s=300, color="#16A34A", edgecolors="white", zorder=6)
        r = datos["resumen"]
        eje.text(0.5, 1.01,
                 f"Inspeccionado {r['cobertura_pct']:.0f}%  Focos {r['focos_confirmados']}/{r['focos_totales']}  "
                 f"Sin detectar {r['infestadas_sin_detectar']}  Puntaje {r['puntaje']:.1f}\n"
                 f"Con falla {r['drones_fallidos']}  Zonas nuevas {r['zonas_nuevas']}  "
                 f"Perdidos {r['mensajes_perdidos']}/{r['mensajes']}  Falsos positivos {r['falsos_positivos']}",
                 transform=eje.transAxes, ha="center", va="bottom", fontsize=8.5, family="monospace")
    leyenda = [Patch(facecolor="#4B5563", label="Zona no sobrevolable"),
               Patch(facecolor="#92400E", label="Zona nueva (dinámica)"),
               Patch(facecolor="#16A34A", label="Estación base"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="#F59E0B", markersize=14, label="Foco confirmado"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="#DC2626", markersize=14, label="Foco sin detectar"),
               Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=11, label="Dron con falla")]
    leyenda += [Line2D([0], [0], color=colores[i], label=f"Dron {i}") for i in range(p.num_drones)]
    fig.legend(handles=leyenda, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("AgroSwarm - Práctica 3: sistema integrado bajo condiciones de robustez", fontsize=12)
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
    fig, ejes = plt.subplots(1, 2, figsize=(13, 7))
    refs = {}
    for eje, (nombre, datos) in zip(ejes, res.items()):
        _preparar_eje(eje, p.tamano_cultivo, nombre)
        if datos["restringidas"]:
            eje.scatter([x[1] for x in datos["restringidas"]], [x[0] for x in datos["restringidas"]],
                        marker="s", s=420, color="#4B5563", zorder=2)
        sc_zonas = eje.scatter([], [], marker="s", s=420, color="#92400E", zorder=2)
        eje.scatter(0, 0, marker="s", s=300, color="#16A34A", zorder=2)
        focos = list(datos["aparicion_foco"])
        sc = eje.scatter([x[1] for x in focos], [x[0] for x in focos], marker="*", s=300, zorder=5)
        lineas = {i: eje.plot([], [], "-", color=colores[i], alpha=0.5)[0] for i in datos["trayectorias"]}
        puntos = {i: eje.scatter([], [], s=170, color=colores[i], edgecolors="white", zorder=6) for i in datos["trayectorias"]}
        txt = eje.text(0.5, 1.01, "", transform=eje.transAxes, ha="center", va="bottom", fontsize=9, family="monospace")
        pasos_falla = {e["id"]: e["paso"] for e in datos["bitacora"] if e["tipo"] == "falla"}
        refs[nombre] = (sc, focos, lineas, puntos, txt, sc_zonas, pasos_falla)
    fig.suptitle("AgroSwarm - Práctica 3: sistema integrado paso a paso (X negra: dron con falla)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def actualizar(t):
        artistas = []
        for nombre, datos in res.items():
            sc, focos, lineas, puntos, txt, sc_zonas, pasos_falla = refs[nombre]
            for i, tr in datos["trayectorias"].items():
                tramo = tr[: t + 1]
                lineas[i].set_data([x[1] for x in tramo], [x[0] for x in tramo])
                puntos[i].set_offsets([[tramo[-1][1], tramo[-1][0]]])
                fallo = i in pasos_falla and pasos_falla[i] <= t
                puntos[i].set_edgecolor("black" if fallo else "white")
                puntos[i].set_linewidth(3 if fallo else 1.5)
            zonas = [z for z, paso in datos["zonas_dinamicas"].items() if paso <= t]
            sc_zonas.set_offsets([[z[1], z[0]] for z in zonas] if zonas else [[float("nan"), float("nan")]])
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
            caidos = sum(1 for paso in pasos_falla.values() if paso <= t)
            txt.set_text(f"Paso {t}/{p.pasos_mision}  Focos activos {sum(c == '#DC2626' for c in cols)}  "
                         f"Drones con falla {caidos}")
            artistas += list(lineas.values()) + list(puntos.values()) + [sc, txt, sc_zonas]
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
    resultados = comparar_mecanismos(parametros)
    imprimir_comparacion(resultados, parametros)
    imprimir_bitacora(resultados)
    imprimir_experimento(experimento_escenarios(parametros), parametros)
    generar_figura(resultados, parametros)
    generar_animacion(resultados, parametros)
