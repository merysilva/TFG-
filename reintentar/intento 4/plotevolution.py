"""
plot_wave_evolution.py
======================
Genera gráficos de área apilada "Evolución de la Onda de Tráfico"
para todos los CSVs de la carpeta que se indique.

Uso:
    python plot_wave_evolution.py                  # busca CSVs en el directorio actual
    python plot_wave_evolution.py ruta/a/resultados  # busca CSVs en esa carpeta

Los PNGs se guardan en la misma carpeta que los CSVs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

# ── Colores y etiquetas ───────────────────────────────────────────────────
STACK_ORDER = ["coches_libre", "coches_ajustando", "coches_atasco", "coches_parados"]

COLORS = {
    "coches_libre":     "#2ECC71",   # verde   — flujo libre
    "coches_ajustando": "#5DADE2",   # azul    — onda de descarga
    "coches_atasco":    "#E74C3C",   # rojo    — núcleo del atasco
    "coches_parados":   "#F39C12",   # naranja — onda de choque
}

LABELS = {
    "coches_libre":     "Crucero (flujo libre)",
    "coches_ajustando": "Ajustando (onda de descarga)",
    "coches_atasco":    "Atasco (núcleo)",
    "coches_parados":   "Frenando (onda de choque)",
}


def friendly_name(filepath):
    name = os.path.basename(filepath).replace("_data.csv", "")
    replacements = {
        "Density_N":              "Densidad N=",
        "Behavior_Aggressive":    "Comportamiento Agresivo",
        "Behavior_Normal":        "Comportamiento Normal",
        "Behavior_Conservative":  "Comportamiento Conservador",
        "Disturbance_D":          "Perturbación D=",
    }
    for k, v in replacements.items():
        if k in name:
            return name.replace(k, v)
    return name


def plot_scenario(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Comprobación de columnas necesarias
    missing = [c for c in STACK_ORDER + ["segundo"] if c not in df.columns]
    if missing:
        print(f"  ⚠️  Saltando {os.path.basename(csv_path)}: faltan columnas {missing}")
        return

    scenario  = friendly_name(csv_path)
    num_cars  = df[STACK_ORDER].iloc[0].sum()
    t         = df["segundo"].values
    stacks    = [df[col].values for col in STACK_ORDER]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.stackplot(
        t, *stacks,
        colors=[COLORS[c] for c in STACK_ORDER],
        labels=[LABELS[c] for c in STACK_ORDER],
        alpha=0.88,
    )

    # Línea de perturbación
    dist_rows = df[df["estado_sistema"] == "FRENADA_ORIGEN"]["segundo"]
    if len(dist_rows) > 0:
        ax.axvline(x=dist_rows.min(), color="black", linestyle="--",
                   lw=1.8, alpha=0.6, label=f"Perturbación (t={dist_rows.min():.0f}s)")

    # Línea de recuperación
    rec_rows = df[df["estado_sistema"] == "RECUPERADO"]["segundo"]
    if len(rec_rows) > 0:
        t_rec = rec_rows.min()
        ax.axvline(x=t_rec, color="#27AE60", linestyle=":", lw=1.8, alpha=0.8,
                   label=f"Recuperación (t={t_rec:.0f}s)")

    # Anotación máximo de atasco
    idx_max = df["coches_atasco"].idxmax()
    max_jam = df.loc[idx_max, "coches_atasco"]
    if max_jam > 1:
        t_max = df.loc[idx_max, "segundo"]
        y_max = df.loc[idx_max, ["coches_parados", "coches_atasco"]].sum()
        ax.annotate(
            f"Máx. atasco\n{max_jam:.0f} veh",
            xy=(t_max, y_max),
            xytext=(t_max + (t[-1] - t[0]) * 0.06, y_max + num_cars * 0.05),
            fontsize=9, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
        )

    ax.set_ylim(0, num_cars * 1.08)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Tiempo de simulación (segundos)", fontsize=12)
    ax.set_ylabel("Número de vehículos", fontsize=12)
    ax.set_title(f"Evolución de la Onda de Tráfico — {scenario}",
                 fontsize=14, fontweight="bold")

    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1], loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()

    out_name = os.path.basename(csv_path).replace("_data.csv", "_wave.png")
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out_name}")


def main():
    # Carpeta de entrada: argumento o directorio actual
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    data_dir = os.path.abspath(data_dir)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"❌ No se encontraron CSVs en: {data_dir}")
        return

    print(f"📊 {len(csv_files)} CSVs encontrados en: {data_dir}")
    print(f"💾 Guardando PNGs en la misma carpeta...\n")

    for f in csv_files:
        plot_scenario(f, data_dir)

    print(f"\n✅ Listo.")


if __name__ == "__main__":
    main()