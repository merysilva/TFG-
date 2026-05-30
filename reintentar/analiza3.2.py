"""
analyze_focused.py — Focused Traffic Jam Analysis for Thesis
=============================================================
Analyzes controlled experiments with clear study-by-study comparisons.

Usage:
    python analyze_focused.py
    
Each individual plot is saved as a separate, labeled file:

Study 1 (Density):
  - study1_density_01_jam_severity.png
  - study1_density_02_recovery_time.png
  - study1_density_03_gap_pressure.png
  - study1_density_04_velocity_drop.png
  - study1_density_05_flow_drop.png
  - study1_density_06_energy_cost.png
  - study1_density_07_timeseries.png

Study 2 (Behavior):
  - study2_behavior_01_jam_severity.png
  - study2_behavior_02_recovery_time.png
  - study2_behavior_03_gap_pressure.png
  - study2_behavior_04_energy_efficiency.png
  - study2_behavior_05_velocity_drop.png
  - study2_behavior_06_flow_capacity.png
  - study2_behavior_07_timeseries.png

Study 3 (Disturbance):
  - study3_disturbance_01_jam_severity.png
  - study3_disturbance_02_recovery_time.png
  - study3_disturbance_03_velocity_drop.png
  - study3_disturbance_04_energy_cost.png
  - study3_disturbance_05_gap_pressure.png
  - study3_disturbance_06_timeseries.png

Combined:
  - combined_01_correlation_heatmap.png
  - combined_02_recovery_by_study.png
  - combined_03_jam_vs_gap.png
  - combined_04_energy_vs_recovery.png
  - combined_05_flow_reduction.png
  - combined_06_summary_table.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = "analysis3"
DATA_DIR = "resultados"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename, title=None):
    """Save a figure with optional suptitle, then close it."""
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    filepath = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {filename}")


def single_ax_fig(figsize=(9, 6)):
    """Return a (fig, ax) pair with a single axes."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ─────────────────────────────────────────────────────────────────
def load_all_scenarios():
    """Load all CSV files from resultados directory."""
    csv_files = list(Path(DATA_DIR).glob("*_data.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {DATA_DIR}/")
        return None

    scenarios = {}
    for filepath in csv_files:
        scenario_name = filepath.stem.replace('_data', '')
        try:
            df = pd.read_csv(filepath)
            scenarios[scenario_name] = df
            print(f"✅ Loaded: {scenario_name} ({len(df)} seconds)")
        except Exception as e:
            print(f"⚠️  Failed to load {scenario_name}: {e}")

    return scenarios


def categorize_scenario(scenario_name):
    """Categorize scenario into study type and extract key parameter."""
    parts = scenario_name.split('_')

    if 'Density' in scenario_name:
        for part in parts:
            if part.startswith('N') and len(part) > 1 and part[1:].isdigit():
                return 'Density', int(part[1:]), scenario_name

    elif 'Behavior' in scenario_name:
        if 'Aggressive' in scenario_name:
            return 'Behavior', 1, scenario_name
        elif 'Normal' in scenario_name:
            return 'Behavior', 2, scenario_name
        elif 'Conservative' in scenario_name:
            return 'Behavior', 3, scenario_name

    elif 'Disturbance' in scenario_name:
        for part in parts:
            if part.startswith('D') and len(part) > 1 and part[1:].isdigit():
                return 'Disturbance', int(part[1:]), scenario_name

    # Fallback
    n_value = None
    for part in parts:
        if part.startswith('N') and len(part) > 1 and part[1:].isdigit():
            n_value = int(part[1:])
            break

    has_s0 = any(part.startswith('S') and len(part) > 1 and
                 part[1:].replace('.', '').isdigit() for part in parts)
    has_t = any(part.startswith('T') and len(part) > 1 and
                part[1:].replace('.', '').isdigit() for part in parts)

    if has_s0 or has_t:
        if 'Safe' in scenario_name or 'Conservative' in scenario_name:
            return 'Behavior', 3, scenario_name
        elif 'Aggressive' in scenario_name or 'HighSpeed' in scenario_name:
            return 'Behavior', 1, scenario_name
        else:
            return 'Behavior', 2, scenario_name
    elif n_value is not None:
        return 'Density', n_value, scenario_name

    return 'Other', 0, scenario_name


def extract_metrics(df, scenario_name):
    """Extract key metrics from scenario dataframe."""
    metrics = {
        'scenario': scenario_name,
        'max_in_jam': df['max_en_atasco'].max(),
        'max_stopped': df['max_parados'].max(),
        'max_gap_pressure': df['presion_gaps'].max(),
        'max_v_diff': df['max_v_diff'].max(),
        'min_velocity': df['vel_min'].min(),
        'avg_velocity_before': df[df['segundo'] < 10]['vel_media'].mean(),
        'avg_velocity_during': df[(df['segundo'] >= 12) & (df['segundo'] <= 20)]['vel_media'].mean(),
        'velocity_drop': 0,
        'total_energy': df['energia_disipada_acum'].max(),
        'total_time_lost': df['tiempo_perdido_acum'].max(),
        'min_efficiency': df['eficiencia_pct'].min(),
        'flow_before': df[df['segundo'] < 10]['flujo_veh_h'].mean(),
        'flow_during': df[(df['segundo'] >= 12) & (df['segundo'] <= 20)]['flujo_veh_h'].mean(),
        'flow_drop_pct': 0,
    }

    metrics['velocity_drop'] = metrics['avg_velocity_before'] - metrics['avg_velocity_during']
    if metrics['flow_before'] > 0:
        metrics['flow_drop_pct'] = (
            100 * (metrics['flow_before'] - metrics['flow_during']) / metrics['flow_before']
        )

    for i in range(15, len(df) - 5):
        window = df.iloc[i:i + 5]
        if (window['eficiencia_pct'].min() > 98 and window['coches_atasco'].max() < 2):
            metrics['recovery_time'] = df.iloc[i]['segundo']
            break
    else:
        metrics['recovery_time'] = np.nan

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  STUDY 1 — DENSITY EFFECTS  (one file per subplot)
# ═══════════════════════════════════════════════════════════════════

def plot_study1_density(density_scenarios):
    print("📊 Generating: Study 1 — Density Effects...")

    data = []
    for n, (scenario_name, df) in sorted(density_scenarios.items()):
        metrics = extract_metrics(df, scenario_name)
        metrics['density'] = n
        data.append(metrics)

    r = pd.DataFrame(data).sort_values('density')

    # 1 — Jam Severity
    fig, ax = single_ax_fig()
    ax.plot(r['density'], r['max_in_jam'], 'o-', lw=3, ms=10, color='#E74C3C')
    ax.set_xlabel('Number of Vehicles (N)')
    ax.set_ylabel('Max Vehicles in Jam')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_01_jam_severity.png',
             'Study 1 — Jam Severity vs Vehicle Density')

    # 2 — Recovery Time
    fig, ax = single_ax_fig()
    recovered = r[r['recovery_time'].notna()]
    if len(recovered) > 0:
        ax.plot(recovered['density'], recovered['recovery_time'],
                'o-', lw=3, ms=10, color='#3498DB')
    ax.set_xlabel('Number of Vehicles (N)')
    ax.set_ylabel('Recovery Time (s)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_02_recovery_time.png',
             'Study 1 — Recovery Duration vs Vehicle Density')

    # 3 — Gap Pressure
    fig, ax = single_ax_fig()
    ax.plot(r['density'], r['max_gap_pressure'], 'o-', lw=3, ms=10, color='#9B59B6')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
    ax.set_xlabel('Number of Vehicles (N)')
    ax.set_ylabel('Max Gap Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_03_gap_pressure.png',
             'Study 1 — Gap Compression vs Vehicle Density')

    # 4 — Velocity Drop
    fig, ax = single_ax_fig()
    ax.plot(r['density'], r['velocity_drop'], 'o-', lw=3, ms=10, color='#E67E22')
    ax.set_xlabel('Number of Vehicles (N)')
    ax.set_ylabel('Velocity Drop (m/s)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_04_velocity_drop.png',
             'Study 1 — Speed Reduction During Jam vs Density')

    # 5 — Flow Capacity Drop
    fig, ax = single_ax_fig()
    ax.plot(r['density'], r['flow_drop_pct'], 'o-', lw=3, ms=10, color='#16A085')
    ax.set_xlabel('Number of Vehicles (N)')
    ax.set_ylabel('Flow Reduction (%)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_05_flow_drop.png',
             'Study 1 — Capacity Drop During Jam vs Density')

    # 6 — Energy Cost
    fig, ax = single_ax_fig()
    ax.plot(r['density'], r['total_energy'], 'o-', lw=3, ms=10, color='#F39C12')
    ax.set_xlabel('Number of Vehicles (N)')
    ax.set_ylabel('Total Energy Lost (J)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_06_energy_cost.png',
             'Study 1 — Energy Cost vs Vehicle Density')

    # 7 — Time Series Comparison (low / mid / high)
    fig, ax = single_ax_fig(figsize=(12, 6))
    densities_to_plot = [r['density'].min(), r['density'].median(), r['density'].max()]
    colors_ts = ['#2ECC71', '#F39C12', '#E74C3C']
    for i, n in enumerate(densities_to_plot):
        n_int = int(n)
        if n_int in density_scenarios:
            sname, df = density_scenarios[n_int]
            ax.plot(df['segundo'], df['coches_atasco'],
                    lw=2.5, label=f'N={n_int}', color=colors_ts[i])
    ax.axvline(x=10, color='black', linestyle='--', lw=2, alpha=0.5, label='Disturbance')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Vehicles in Jam State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study1_density_07_timeseries.png',
             'Study 1 — Jam Evolution: Low vs Medium vs High Density')

    r.to_csv(f"{OUTPUT_DIR}/study1_density_data.csv", index=False)
    print("  ✅ Saved: study1_density_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  STUDY 2 — BEHAVIOR OPTIMIZATION  (one file per subplot)
# ═══════════════════════════════════════════════════════════════════

def plot_study2_behavior(behavior_scenarios):
    print("📊 Generating: Study 2 — Behavior Optimization...")

    behavior_order = ['Aggressive', 'Normal', 'Conservative']
    behavior_labels = {
        'Aggressive':   'Aggressive\n(s₀=2.0, T=0.8)',
        'Normal':       'Normal\n(s₀=2.5, T=1.0)',
        'Conservative': 'Conservative\n(s₀=4.0, T=1.5)',
    }
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    data = []
    for btype in behavior_order:
        if btype in behavior_scenarios:
            sname, df = behavior_scenarios[btype]
            metrics = extract_metrics(df, sname)
            metrics['behavior'] = btype
            data.append(metrics)

    r = pd.DataFrame(data)
    x_pos = np.arange(len(behavior_order))
    tick_labels = [behavior_labels[b] for b in behavior_order]

    def bar_with_labels(ax, values, fmt='{:.0f}', suffix=''):
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    fmt.format(h) + suffix,
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # 1 — Jam Severity
    fig, ax = single_ax_fig()
    bar_with_labels(ax, r['max_in_jam'])
    ax.set_ylabel('Max Vehicles in Jam')
    save_fig(fig, 'study2_behavior_01_jam_severity.png',
             'Study 2 — Jam Severity by Driver Behavior')

    # 2 — Recovery Time
    fig, ax = single_ax_fig()
    recovery_vals = [
        r[r['behavior'] == b]['recovery_time'].values[0]
        for b in behavior_order
        if b in r['behavior'].values
    ]
    if len(recovery_vals) == len(behavior_order) and not any(np.isnan(recovery_vals)):
        bar_with_labels(ax, recovery_vals, fmt='{:.0f}', suffix='s')
    ax.set_ylabel('Recovery Time (s)')
    save_fig(fig, 'study2_behavior_02_recovery_time.png',
             'Study 2 — Recovery Speed by Driver Behavior')

    # 3 — Gap Pressure
    fig, ax = single_ax_fig()
    bar_with_labels(ax, r['max_gap_pressure'], fmt='{:.2f}')
    ax.axhline(y=1.0, color='red', linestyle='--', lw=2, alpha=0.7, label='Critical')
    ax.set_ylabel('Max Gap Pressure')
    ax.legend()
    save_fig(fig, 'study2_behavior_03_gap_pressure.png',
             'Study 2 — Gap Management Quality by Driver Behavior')

    # 4 — Energy Efficiency
    fig, ax = single_ax_fig()
    bar_with_labels(ax, r['total_energy'], fmt='{:.0f}')
    ax.set_ylabel('Total Energy Lost (J)')
    save_fig(fig, 'study2_behavior_04_energy_efficiency.png',
             'Study 2 — Energy Efficiency by Driver Behavior')

    # 5 — Velocity Drop
    fig, ax = single_ax_fig()
    bar_with_labels(ax, r['velocity_drop'], fmt='{:.2f}')
    ax.set_ylabel('Velocity Drop (m/s)')
    save_fig(fig, 'study2_behavior_05_velocity_drop.png',
             'Study 2 — Speed Degradation by Driver Behavior')

    # 6 — Flow Capacity
    fig, ax = single_ax_fig()
    bar_with_labels(ax, r['flow_drop_pct'], fmt='{:.1f}', suffix='%')
    ax.set_ylabel('Flow Reduction (%)')
    save_fig(fig, 'study2_behavior_06_flow_capacity.png',
             'Study 2 — Capacity Impact by Driver Behavior')

    # 7 — Time Series
    fig, ax = single_ax_fig(figsize=(12, 6))
    for i, btype in enumerate(behavior_order):
        if btype in behavior_scenarios:
            sname, df = behavior_scenarios[btype]
            ax.plot(df['segundo'], df['coches_atasco'],
                    lw=3, label=behavior_labels[btype].replace('\n', ' '),
                    color=colors[i])
    ax.axvline(x=10, color='black', linestyle='--', lw=2, alpha=0.5, label='Disturbance')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Vehicles in Jam State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study2_behavior_07_timeseries.png',
             'Study 2 — Jam Evolution: Behavior Comparison')

    r.to_csv(f"{OUTPUT_DIR}/study2_behavior_data.csv", index=False)
    print("  ✅ Saved: study2_behavior_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  STUDY 3 — DISTURBANCE SENSITIVITY  (one file per subplot)
# ═══════════════════════════════════════════════════════════════════

def plot_study3_disturbance(disturbance_scenarios):
    print("📊 Generating: Study 3 — Disturbance Sensitivity...")

    data = []
    for d_value, (sname, df) in sorted(disturbance_scenarios.items()):
        metrics = extract_metrics(df, sname)
        metrics['disturbance'] = d_value
        data.append(metrics)

    r = pd.DataFrame(data).sort_values('disturbance')

    # 1 — Jam Severity
    fig, ax = single_ax_fig()
    ax.plot(r['disturbance'], r['max_in_jam'], 'o-', lw=3, ms=10, color='#E74C3C')
    ax.set_xlabel('Disturbance Strength (m/s²)')
    ax.set_ylabel('Max Vehicles in Jam')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study3_disturbance_01_jam_severity.png',
             'Study 3 — Jam Severity vs Disturbance Strength')

    # 2 — Recovery Time
    fig, ax = single_ax_fig()
    recovered = r[r['recovery_time'].notna()]
    if len(recovered) > 0:
        ax.plot(recovered['disturbance'], recovered['recovery_time'],
                'o-', lw=3, ms=10, color='#3498DB')
    ax.set_xlabel('Disturbance Strength (m/s²)')
    ax.set_ylabel('Recovery Time (s)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study3_disturbance_02_recovery_time.png',
             'Study 3 — Recovery Duration vs Disturbance Strength')

    # 3 — Velocity Drop
    fig, ax = single_ax_fig()
    ax.plot(r['disturbance'], r['velocity_drop'], 'o-', lw=3, ms=10, color='#E67E22')
    ax.set_xlabel('Disturbance Strength (m/s²)')
    ax.set_ylabel('Velocity Drop (m/s)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study3_disturbance_03_velocity_drop.png',
             'Study 3 — Speed Impact vs Disturbance Strength')

    # 4 — Energy Cost
    fig, ax = single_ax_fig()
    ax.plot(r['disturbance'], r['total_energy'], 'o-', lw=3, ms=10, color='#9B59B6')
    ax.set_xlabel('Disturbance Strength (m/s²)')
    ax.set_ylabel('Total Energy Lost (J)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study3_disturbance_04_energy_cost.png',
             'Study 3 — Energy Cost vs Disturbance Strength')

    # 5 — Gap Pressure
    fig, ax = single_ax_fig()
    ax.plot(r['disturbance'], r['max_gap_pressure'], 'o-', lw=3, ms=10, color='#1ABC9C')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
    ax.set_xlabel('Disturbance Strength (m/s²)')
    ax.set_ylabel('Max Gap Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study3_disturbance_05_gap_pressure.png',
             'Study 3 — Gap Compression vs Disturbance Strength')

    # 6 — Time Series (all disturbances)
    fig, ax = single_ax_fig(figsize=(12, 6))
    n_scenarios = len(disturbance_scenarios)
    colors_cmap = plt.cm.Reds(np.linspace(0.3, 0.9, n_scenarios))
    for i, (d_value, (sname, df)) in enumerate(sorted(disturbance_scenarios.items())):
        ax.plot(df['segundo'], df['coches_atasco'],
                lw=2.5, label=f'D={d_value}', color=colors_cmap[i])
    ax.axvline(x=10, color='black', linestyle='--', lw=2, alpha=0.5, label='Disturbance')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Vehicles in Jam State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'study3_disturbance_06_timeseries.png',
             'Study 3 — Jam Evolution by Disturbance Strength')

    r.to_csv(f"{OUTPUT_DIR}/study3_disturbance_data.csv", index=False)
    print("  ✅ Saved: study3_disturbance_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  COMBINED ANALYSIS  (one file per subplot)
# ═══════════════════════════════════════════════════════════════════

def plot_combined_insights(all_scenarios):
    print("📊 Generating: Combined Insights...")

    all_data = []
    for name, df in all_scenarios.items():
        study_type, param_value, _ = categorize_scenario(name)
        if study_type != 'Other':
            metrics = extract_metrics(df, name)
            metrics['study'] = study_type
            metrics['param_value'] = param_value
            all_data.append(metrics)

    combined_df = pd.DataFrame(all_data)
    colors_study = {'Density': '#E74C3C', 'Behavior': '#3498DB', 'Disturbance': '#2ECC71'}

    # ---- 1 — Correlation Heatmap ----
    corr_columns = ['max_in_jam', 'recovery_time', 'max_gap_pressure',
                    'velocity_drop', 'total_energy', 'flow_drop_pct']
    corr_data = combined_df[corr_columns].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    save_fig(fig, 'combined_01_correlation_heatmap.png',
             'Combined — Correlation Matrix: Key Jam Metrics')

    # ---- 2 — Recovery Time by Study ----
    fig, ax = single_ax_fig()
    recovered = combined_df[combined_df['recovery_time'].notna()]
    if len(recovered) > 0:
        study_types = recovered['study'].unique()
        bp = ax.boxplot(
            [recovered[recovered['study'] == st]['recovery_time'].values for st in study_types],
            labels=study_types, patch_artist=True
        )
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
    ax.set_ylabel('Recovery Time (s)')
    ax.grid(True, alpha=0.3, axis='y')
    save_fig(fig, 'combined_02_recovery_by_study.png',
             'Combined — Recovery Time Distribution by Study')

    # ---- 3 — Jam Severity vs Gap Pressure ----
    fig, ax = single_ax_fig()
    for study in combined_df['study'].unique():
        subset = combined_df[combined_df['study'] == study]
        ax.scatter(subset['max_gap_pressure'], subset['max_in_jam'],
                   s=150, alpha=0.7, label=study, color=colors_study.get(study, 'gray'))
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax.set_xlabel('Max Gap Pressure')
    ax.set_ylabel('Max Vehicles in Jam')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'combined_03_jam_vs_gap.png',
             'Combined — Jam Severity vs Gap Compression (All Studies)')

    # ---- 4 — Energy vs Recovery Time ----
    fig, ax = single_ax_fig()
    recovered = combined_df[combined_df['recovery_time'].notna()]
    if len(recovered) > 0:
        for study in recovered['study'].unique():
            subset = recovered[recovered['study'] == study]
            ax.scatter(subset['recovery_time'], subset['total_energy'],
                       s=150, alpha=0.7, label=study, color=colors_study.get(study, 'gray'))
    ax.set_xlabel('Recovery Time (s)')
    ax.set_ylabel('Total Energy Lost (J)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'combined_04_energy_vs_recovery.png',
             'Combined — Energy Cost vs Recovery Duration')

    # ---- 5 — Flow Reduction by Study ----
    fig, ax = single_ax_fig()
    study_means = combined_df.groupby('study')['flow_drop_pct'].mean()
    bars = ax.bar(range(len(study_means)), study_means.values,
                  color=[colors_study.get(s, 'gray') for s in study_means.index])
    ax.set_xticks(range(len(study_means)))
    ax.set_xticklabels(study_means.index, fontsize=11)
    ax.set_ylabel('Avg Flow Reduction (%)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    save_fig(fig, 'combined_05_flow_reduction.png',
             'Combined — Average Capacity Drop by Study')

    # ---- 6 — Summary Table ----
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    lines = ["COMBINED ANALYSIS SUMMARY", ""]
    for study in ['Density', 'Behavior', 'Disturbance']:
        subset = combined_df[combined_df['study'] == study]
        if len(subset) == 0:
            continue
        lines.append(f"{study.upper()} STUDY  ({len(subset)} scenarios)")
        lines.append(f"  Max in Jam:      {subset['max_in_jam'].min():.0f} – {subset['max_in_jam'].max():.0f} vehicles  (avg {subset['max_in_jam'].mean():.1f})")
        rec = subset[subset['recovery_time'].notna()]
        if len(rec) > 0:
            lines.append(f"  Recovery Time:   {rec['recovery_time'].min():.0f} – {rec['recovery_time'].max():.0f} s  (avg {rec['recovery_time'].mean():.1f})")
        else:
            lines.append("  Recovery Time:   No recovery within simulation time")
        lines.append(f"  Gap Pressure:    {subset['max_gap_pressure'].min():.2f} – {subset['max_gap_pressure'].max():.2f}  (avg {subset['max_gap_pressure'].mean():.2f})")
        lines.append(f"  Velocity Drop:   {subset['velocity_drop'].min():.1f} – {subset['velocity_drop'].max():.1f} m/s  (avg {subset['velocity_drop'].mean():.1f})")
        lines.append(f"  Energy Lost:     {subset['total_energy'].min():.0f} – {subset['total_energy'].max():.0f} J  (avg {subset['total_energy'].mean():.0f})")
        lines.append(f"  Flow Reduction:  {subset['flow_drop_pct'].min():.1f} – {subset['flow_drop_pct'].max():.1f}%  (avg {subset['flow_drop_pct'].mean():.1f}%)")
        lines.append("")

    lines.append("KEY FINDINGS")
    best_recovery = combined_df[combined_df['recovery_time'].notna()].nsmallest(1, 'recovery_time')
    worst_jam = combined_df.nlargest(1, 'max_in_jam')
    least_energy = combined_df.nsmallest(1, 'total_energy')
    if len(best_recovery):
        lines.append(f"  Fastest Recovery :  {best_recovery.iloc[0]['scenario']}  ({best_recovery.iloc[0]['recovery_time']:.0f} s)")
    if len(worst_jam):
        lines.append(f"  Most Severe Jam  :  {worst_jam.iloc[0]['scenario']}  ({worst_jam.iloc[0]['max_in_jam']:.0f} vehicles)")
    if len(least_energy):
        lines.append(f"  Most Efficient   :  {least_energy.iloc[0]['scenario']}  ({least_energy.iloc[0]['total_energy']:.0f} J)")
    lines.append(f"  Strongest Corr.  :  max_gap_pressure ↔ max_in_jam  (r = {corr_data.loc['max_gap_pressure', 'max_in_jam']:.2f})")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    save_fig(fig, 'combined_06_summary_table.png',
             'Combined — Analysis Summary')

    combined_df.to_csv(f"{OUTPUT_DIR}/combined_data.csv", index=False)
    print("  ✅ Saved: combined_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "FOCUSED TRAFFIC JAM ANALYSIS" + " " * 33 + "║")
    print("║" + " " * 22 + "Thesis-Ready Plots" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝\n")

    scenarios = load_all_scenarios()
    if not scenarios:
        print(f"\n❌ No scenario data found in {DATA_DIR}/\n")
        return

    print(f"\n📊 Loaded {len(scenarios)} scenarios")
    print(f"📁 Saving plots to: {OUTPUT_DIR}/\n")

    density_scenarios = {}
    behavior_scenarios = {}
    disturbance_scenarios = {}

    for name, df in scenarios.items():
        study_type, param_value, _ = categorize_scenario(name)
        if study_type == 'Density':
            density_scenarios[param_value] = (name, df)
        elif study_type == 'Behavior':
            bname = 'Aggressive' if param_value == 1 else ('Normal' if param_value == 2 else 'Conservative')
            behavior_scenarios[bname] = (name, df)
        elif study_type == 'Disturbance':
            disturbance_scenarios[param_value] = (name, df)

    print("=" * 80)
    print("STUDY-SPECIFIC ANALYSES")
    print("=" * 80 + "\n")

    if len(density_scenarios) >= 3:
        plot_study1_density(density_scenarios)
    else:
        print("  ⚠️  Skipping Study 1: need at least 3 density scenarios")

    if len(behavior_scenarios) >= 2:
        plot_study2_behavior(behavior_scenarios)
    else:
        print("  ⚠️  Skipping Study 2: need at least 2 behavior scenarios")

    if len(disturbance_scenarios) >= 3:
        plot_study3_disturbance(disturbance_scenarios)
    else:
        print("  ⚠️  Skipping Study 3: need at least 3 disturbance scenarios")

    print("\n" + "=" * 80)
    print("COMBINED ANALYSIS")
    print("=" * 80 + "\n")
    plot_combined_insights(scenarios)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📊 Output directory: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(Path(OUTPUT_DIR).glob("*.png")):
        print(f"  • {f.name}")
    print()


if __name__ == "__main__":
    main()
