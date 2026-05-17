"""
simulador.py  —  TFG Stop-and-Go  (versión simple)
====================================================
- Corre todos los escenarios en modo headless (sin ventana)
- Guarda resultados_master.csv con una fila por experimento

Parámetros variados:
    v_crucero, t_frenada, num_cars, t_reaccion, a_brake

Salidas por experimento:
    t_dissolve      — segundos hasta que el atasco se disipa
    max_en_atasco   — pico de coches parados simultáneamente
"""

import math
import csv
import itertools

# ── Parámetros fijos ──────────────────────────────────────────
TRACK_RADIUS = 400
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS
S0           = 5.0
ACC_MAX      = 1.2
DEC_MAX      = 2.0
DT           = 0.1          # paso de tiempo (s)
MAX_TIME     = 400          # cortafuegos: atasco perpetuo

# ── Rejilla de experimentos ───────────────────────────────────
V_CRUCEROS   = [12.0, 20.0, 28.0]      # m/s
T_FRENADAS   = [3.0, 6.0, 10.0]        # s
NUMS_CARS    = [25, 38, 55]
T_REACCIONES = [1.0, 1.5, 2.5]         # s
A_BRAKES     = [-6.0, -10.0, -14.0]    # m/s²

# Producto cartesiano = 3^5 = 243 experimentos
ESCENARIOS = [
    {"v_crucero": v, "t_frenada": tf, "num_cars": nc,
     "t_reaccion": tr, "a_brake": ab}
    for v, tf, nc, tr, ab in itertools.product(
        V_CRUCEROS, T_FRENADAS, NUMS_CARS, T_REACCIONES, A_BRAKES)
]


def correr_experimento(cfg):
    v0      = cfg["v_crucero"]
    n       = cfg["num_cars"]
    tr      = cfg["t_reaccion"]
    ab      = cfg["a_brake"]
    tf_dur  = cfg["t_frenada"]

    pos = [i * (TRACK_LENGTH / n) for i in range(n)]
    vel = [v0] * n

    estado         = "ESTABILIZANDO"
    timer_frenada  = 0.0
    t              = 0.0
    t_dissolve     = None
    max_en_atasco  = 0

    while t < MAX_TIME:
        # Transicion ESTABILIZANDO -> FRENANDO a t=7s
        if estado == "ESTABILIZANDO" and t >= 7.0:
            estado        = "FRENANDO"
            timer_frenada = tf_dur

        # Calcular aceleraciones IDM
        acc = [0.0] * n
        if estado != "ESTABILIZANDO":
            for i in range(n):
                lead = (i + 1) % n
                s    = max(0.1, (pos[lead] - pos[i]) % TRACK_LENGTH - 5.0)
                v    = vel[i]
                dv   = v - vel[lead]
                s_st = S0 + max(0, v * tr + v * dv / (2 * math.sqrt(ACC_MAX * DEC_MAX)))
                acc[i] = ACC_MAX * (1 - (v / v0) ** 4 - (s_st / s) ** 2)

        # Coche 0 frena durante t_frenada
        if estado == "FRENANDO":
            acc[0]         = ab
            timer_frenada -= DT
            if timer_frenada <= 0:
                estado = "RECUPERANDO"

        # Integrar
        for i in range(n):
            vel[i] = max(0.0, vel[i] + acc[i] * DT)
            pos[i] = (pos[i] + vel[i] * DT) % TRACK_LENGTH

        t = round(t + DT, 6)

        # Metricas
        if estado in ("FRENANDO", "RECUPERANDO"):
            parados = sum(1 for v in vel if v < v0 / 2)
            if parados > max_en_atasco:
                max_en_atasco = parados

            # Disipado: todos los coches cerca de v0
            if estado == "RECUPERANDO" and all(v >= v0 - 1.0 for v in vel):
                t_dissolve = round(t, 1)
                break

    return {
        "v_crucero":     cfg["v_crucero"],
        "t_frenada":     cfg["t_frenada"],
        "num_cars":      cfg["num_cars"],
        "t_reaccion":    cfg["t_reaccion"],
        "a_brake":       cfg["a_brake"],
        "t_dissolve":    t_dissolve if t_dissolve is not None else "NaN",
        "max_en_atasco": max_en_atasco,
    }


# ── Ejecutar todos y guardar ──────────────────────────────────
total = len(ESCENARIOS)
print(f"\nCorriendo {total} experimentos...\n")

resultados = []
for i, cfg in enumerate(ESCENARIOS, 1):
    r = correr_experimento(cfg)
    resultados.append(r)
    print(f"  [{i:03d}/{total}]  v={cfg['v_crucero']}  tf={cfg['t_frenada']}  "
          f"N={cfg['num_cars']}  Tr={cfg['t_reaccion']}  ab={cfg['a_brake']}  "
          f"->  T_dissolve={r['t_dissolve']}s  max_atasco={r['max_en_atasco']}")

with open("resultados_master.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
    writer.writeheader()
    writer.writerows(resultados)

print(f"\nListo. resultados_master.csv generado con {total} filas.\n")
