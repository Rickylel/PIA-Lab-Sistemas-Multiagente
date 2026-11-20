# AgroSwarm: Sistema Multiagente de Drones para la Detección Temprana de Plagas en Cultivos

Portafolio del Laboratorio de Sistemas Multiagente (PIA). Una flota de drones inspecciona un cultivo dividido en parcelas para encontrar focos de plaga antes de que se propaguen.

## Características del problema

- La plaga **se propaga** a parcelas vecinas si nadie la detecta.
- Los drones tienen **batería limitada** y deben regresar a la estación base a recargar; un dron que se queda sin batería en el campo se pierde.
- La cámara ve una ventana de 3x3 parcelas; los síntomas vistos desde lejos son **sospechas** que se **confirman sobrevolando** la parcela. Un foco confirmado se registra en el mapa de infestación y se manda a fumigar.
- Hay **zonas no sobrevolables** (árboles altos, líneas eléctricas) y, en la Práctica 3, zonas que se vuelven no sobrevolables durante la misión.

## Estructura

| Archivo | Fase | Qué compara |
|---|---|---|
| `practica1_dron_explorador.py` | Fase 1: fundamentos de agentes | Un dron reactivo frente a uno basado en modelos |
| `practica2_coordinacion_drones.py` | Fase 2: coordinación | Pizarrón compartido (estación base) frente a mensajería directa, con ambas arquitecturas |
| `practica3_sistema_drones_plagas.py` | Fase 3: sistema integrado | Ambos mecanismos con falla de drones, pérdida de enlace, zonas nuevas y sensor con error |

Cada script genera un reporte en la terminal, una figura (`comparacion_practicaN.png`) y una animación (`simulacion_practicaN.gif`). Los archivos `*_salida.txt` contienen el reporte de la última ejecución.

## Ejecución

```bash
pip install -r requirements.txt
python practica1_dron_explorador.py
python practica2_coordinacion_drones.py
python practica3_sistema_drones_plagas.py
```

Requiere Python 3.9 o superior. Los parámetros de cada experimento se modifican en la Sección 0 (`ParametrosPractica`) de cada script.
