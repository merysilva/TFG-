"""
plot_wave_evolution.py
======================
Generates stacked area plots "Traffic Wave Evolution"
for all CSVs in the specified folder.

Usage:
    python plot_wave_evolution.py                    # looks for CSVs in the current directory
    python plot_wave_evolution.py path/to/results    # looks for CSVs in that folder

PNGs are saved in the same folder as the CSVs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

# ── Colours and labels ────────────────────────────────────────────────────
STACK_ORDER = ["coches_libre", "coches_ajustando", "coches_atasco", "coches_parados"]

COLORS = {
    "coches_libre":     "#2ECC71",   # green  — free flow
    "coches_ajustando": "#5DADE2",   # blue   — discharge wave
    "coches_atasco":    "#E74C3C",   # red    — jam core
    "coches_parados":   "#F39C12",   # orange — shock wave
}

LABELS = {
    "coches_libre":     "Cruising (free flow)",
    "coches_ajustando": "Adjusting (discharge wave)",
    "coches_atasco":    "Congested (jam core)",
    "coches_parados":   "Braking (shock wave)",
}


def friendly_name(filepath):
    name = os.path.basename(filepath).replace("_data.csv", "")
    replacements = {
        "Density_N":              "Density N=",
        "Behavior_Aggressive":    "Aggressive Behaviour",
        "Behavior_Normal":        "Normal Behaviour",
        "Behavior_Conservative":  "Conservative Behaviour",
        "Disturbance_D":          "Disturbance D=",
    }
    for k, v in replacements.items():
        if k in name:
            return name.replace(k, v)
    return name


def plot_scenario(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Check for required columns
    missing = [c for c in STACK_ORDER + ["segundo"] if c not in df.columns]
    if missing:
        print(f"  ⚠️  Skipping {os.path.basename(csv_path)}: missing columns {missing}")
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

    # Disturbance line
    dist_rows = df[df["estado_sistema"] == "FRENADA_ORIGEN"]["segundo"]
    if len(dist_rows) > 0:
        ax.axvline(x=dist_rows.min(), color="black", linestyle="--",
                   lw=1.8, alpha=0.6, label=f"Disturbance (t={dist_rows.min():.0f}s)")

    # Recovery line
    rec_rows = df[df["estado_sistema"] == "RECUPERADO"]["segundo"]
    if len(rec_rows) > 0:
        t_rec = rec_rows.min()
        ax.axvline(x=t_rec, color="#27AE60", linestyle=":", lw=1.8, alpha=0.8,
                   label=f"Recovery (t={t_rec:.0f}s)")

    # Peak congestion annotation
    idx_max = df["coches_atasco"].idxmax()
    max_jam = df.loc[idx_max, "coches_atasco"]
    if max_jam > 1:
        t_max = df.loc[idx_max, "segundo"]
        y_max = df.loc[idx_max, ["coches_parados", "coches_atasco"]].sum()
        ax.annotate(
            f"Peak jam\n{max_jam:.0f} veh",
            xy=(t_max, y_max),
            xytext=(t_max + (t[-1] - t[0]) * 0.06, y_max + num_cars * 0.05),
            fontsize=9, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
        )

    ax.set_ylim(0, num_cars * 1.08)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Simulation time (seconds)", fontsize=12)
    ax.set_ylabel("Number of vehicles", fontsize=12)
    ax.set_title(f"Traffic Wave Evolution — {scenario}",
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
    # Input folder: argument or current directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    data_dir = os.path.abspath(data_dir)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"❌ No CSVs found in: {data_dir}")
        return

    print(f"📊 {len(csv_files)} CSVs found in: {data_dir}")
    print(f"💾 Saving PNGs in the same folder...\n")

    for f in csv_files:
        plot_scenario(f, data_dir)

    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()