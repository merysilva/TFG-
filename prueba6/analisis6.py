"""
analyze_traffic_experiments.py — Comprehensive Analysis for All Simulations
===========================================================================
Generates plots and statistical analysis for both:
    1. Homogeneous experiments (density × speed)
    2. Heterogeneous experiments (trucks × aggression)

Usage:
    python analyze_traffic_experiments.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from scipy import stats

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = "analysis_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(filename):
    """Load master CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded: {filename} ({len(df)} scenarios)")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found")
        return None


def load_scenario_data(scenario_name, folder):
    """Load time-series data for a specific scenario."""
    filepath = f"{folder}/{scenario_name}_datos.csv"
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None


# ═══════════════════════════════════════════════════════════════════
#  HOMOGENEOUS ANALYSIS (Density × Speed)
# ═══════════════════════════════════════════════════════════════════

def plot_homogeneous_heatmap(df):
    """Heatmap: Dissolution time by density and speed."""
    print("Generating: Homogeneous dissolution time heatmap...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    if df['t_dissolve_numeric'].isna().all():
        print("  ⚠️  SKIP: All dissolution times are NaN (perpetual jams)")
        return
    
    pivot = df.pivot_table(
        values='t_dissolve_numeric',
        index='num_cars',
        columns='max_speed',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Dissolution Time (seconds)'},
                linewidths=0.5)
    plt.xlabel('Maximum Speed (m/s)', fontsize=12)
    plt.ylabel('Number of Vehicles (Density)', fontsize=12)
    plt.title('Traffic Jam Dissolution Time: Homogeneous Flow\n(Lower is better)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/homo_01_dissolution_heatmap.png", dpi=300)
    plt.close()
    print("  ✅ Saved: homo_01_dissolution_heatmap.png")


def plot_homogeneous_trends(df):
    """Line plots: Effects of density and speed."""
    print("Generating: Homogeneous trend analysis...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Effect of density at different speeds
    for speed in sorted(df['max_speed'].unique()):
        subset = df[df['max_speed'] == speed]
        axes[0].plot(subset['num_cars'], subset['t_dissolve_numeric'], 
                    marker='o', linewidth=2, markersize=8, label=f'{int(speed)} m/s')
    
    axes[0].set_xlabel('Number of Vehicles', fontsize=12)
    axes[0].set_ylabel('Dissolution Time (s)', fontsize=12)
    axes[0].set_title('Effect of Vehicle Density', fontsize=13, fontweight='bold')
    axes[0].legend(title='Max Speed', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Effect of speed at different densities
    for density in sorted(df['num_cars'].unique()):
        subset = df[df['num_cars'] == density]
        axes[1].plot(subset['max_speed'], subset['t_dissolve_numeric'], 
                    marker='s', linewidth=2, markersize=8, label=f'{density} vehicles')
    
    axes[1].set_xlabel('Maximum Speed (m/s)', fontsize=12)
    axes[1].set_ylabel('Dissolution Time (s)', fontsize=12)
    axes[1].set_title('Effect of Maximum Speed', fontsize=13, fontweight='bold')
    axes[1].legend(title='Density', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Homogeneous Traffic: Density and Speed Effects', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/homo_02_trends.png", dpi=300)
    plt.close()
    print("  ✅ Saved: homo_02_trends.png")


def plot_homogeneous_time_series(df_master):
    """Time-series comparison for selected homogeneous scenarios."""
    print("Generating: Homogeneous time-series comparison...")
    
    # Select representative scenarios
    scenarios = [
        'Homo_N20_V30',  # Low density, high speed
        'Homo_N27_V30',  # Medium density, high speed
        'Homo_N35_V30',  # High density, high speed
        'Homo_N27_V20',  # Medium density, low speed
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for scenario_name in scenarios:
        df = load_scenario_data(scenario_name, "resultados_homogeneo")
        if df is not None:
            axes[0, 0].plot(df['segundo'], df['avg_velocity'], linewidth=2, label=scenario_name)
            axes[0, 1].plot(df['segundo'], df['vehicles_stopped'], linewidth=2, label=scenario_name)
            axes[1, 0].plot(df['segundo'], df['v_diff'], linewidth=2, label=scenario_name)
            axes[1, 1].plot(df['segundo'], df['avg_gap'], linewidth=2, label=scenario_name)
    
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_ylabel('Average Velocity (m/s)', fontsize=11)
    axes[0, 0].set_title('Average Velocity Evolution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Stopped Vehicles', fontsize=11)
    axes[0, 1].set_title('Stopped Vehicles Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('V_diff (m/s)', fontsize=11)
    axes[1, 0].set_title('Speed Variance (V_diff)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Average Gap (m)', fontsize=11)
    axes[1, 1].set_title('Average Following Distance', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    plt.suptitle('Homogeneous Traffic: Time-Series Comparison', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/homo_03_time_series.png", dpi=300)
    plt.close()
    print("  ✅ Saved: homo_03_time_series.png")


# ═══════════════════════════════════════════════════════════════════
#  HETEROGENEOUS ANALYSIS (Trucks × Aggression)
# ═══════════════════════════════════════════════════════════════════

def plot_heterogeneous_heatmap(df):
    """Heatmap: Dissolution time by truck % and aggression %."""
    print("Generating: Heterogeneous dissolution time heatmap...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    if df['t_dissolve_numeric'].isna().all():
        print("  ⚠️  SKIP: All dissolution times are NaN (perpetual jams)")
        return
    
    pivot = df.pivot_table(
        values='t_dissolve_numeric',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    
    pivot.index = (pivot.index * 100).astype(int)
    pivot.columns = (pivot.columns * 100).astype(int)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Dissolution Time (seconds)'},
                linewidths=0.5)
    plt.xlabel('Aggressive Drivers (%)', fontsize=12)
    plt.ylabel('Trucks (%)', fontsize=12)
    plt.title('Traffic Jam Dissolution Time: Heterogeneous Flow\n(Lower is better)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/het_01_dissolution_heatmap.png", dpi=300)
    plt.close()
    print("  ✅ Saved: het_01_dissolution_heatmap.png")


def plot_heterogeneous_trends(df):
    """Line plots: Effects of truck % and aggression %."""
    print("Generating: Heterogeneous trend analysis...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    df['truck_pct_label'] = (df['truck_pct'] * 100).astype(int)
    df['aggressive_pct_label'] = (df['aggressive_pct'] * 100).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Effect of truck % at different aggression levels
    for aggr in sorted(df['aggressive_pct'].unique()):
        subset = df[df['aggressive_pct'] == aggr]
        aggr_label = int(aggr * 100)
        axes[0].plot(subset['truck_pct_label'], subset['t_dissolve_numeric'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f'{aggr_label}% Aggressive')
    
    axes[0].set_xlabel('Truck Percentage (%)', fontsize=12)
    axes[0].set_ylabel('Dissolution Time (s)', fontsize=12)
    axes[0].set_title('Effect of Truck Percentage', fontsize=13, fontweight='bold')
    axes[0].legend(title='Driver Aggression', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Effect of aggression at different truck levels
    for truck in sorted(df['truck_pct'].unique()):
        subset = df[df['truck_pct'] == truck]
        truck_label = int(truck * 100)
        axes[1].plot(subset['aggressive_pct_label'], subset['t_dissolve_numeric'], 
                    marker='s', linewidth=2, markersize=8, 
                    label=f'{truck_label}% Trucks')
    
    axes[1].set_xlabel('Aggressive Driver Percentage (%)', fontsize=12)
    axes[1].set_ylabel('Dissolution Time (s)', fontsize=12)
    axes[1].set_title('Effect of Driver Aggression', fontsize=13, fontweight='bold')
    axes[1].legend(title='Fleet Composition', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Heterogeneous Traffic: Composition Effects', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/het_02_trends.png", dpi=300)
    plt.close()
    print("  ✅ Saved: het_02_trends.png")


def plot_heterogeneous_time_series(df_master):
    """Time-series comparison for selected heterogeneous scenarios."""
    print("Generating: Heterogeneous time-series comparison...")
    
    # Select representative scenarios
    scenarios = [
        'Het_T00_A000',  # Baseline: all cars, all cautious
        'Het_T00_A100',  # All cars, all aggressive
        'Het_T25_A050',  # 25% trucks, mixed drivers
        'Het_T50_A000',  # 50% trucks, all cautious
        'Het_T75_A050',  # 75% trucks, mixed drivers
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for scenario_name in scenarios:
        df = load_scenario_data(scenario_name, "resultados_heterogeneo_final")
        if df is not None:
            axes[0, 0].plot(df['segundo'], df['avg_velocity'], linewidth=2, label=scenario_name)
            axes[0, 1].plot(df['segundo'], df['vehicles_stopped'], linewidth=2, label=scenario_name)
            axes[1, 0].plot(df['segundo'], df['v_diff'], linewidth=2, label=scenario_name)
            
            # Cars vs trucks velocity
            axes[1, 1].plot(df['segundo'], df['car_mean_velocity'], 
                          linewidth=2, linestyle='-', alpha=0.7, label=f'{scenario_name} (Cars)')
            axes[1, 1].plot(df['segundo'], df['truck_mean_velocity'], 
                          linewidth=2, linestyle='--', alpha=0.7, label=f'{scenario_name} (Trucks)')
    
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_ylabel('Average Velocity (m/s)', fontsize=11)
    axes[0, 0].set_title('System Average Velocity', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Stopped Vehicles', fontsize=11)
    axes[0, 1].set_title('Stopped Vehicles Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('V_diff (m/s)', fontsize=11)
    axes[1, 0].set_title('Speed Variance (V_diff)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Velocity (m/s)', fontsize=11)
    axes[1, 1].set_title('Cars vs Trucks Speed Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=7, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=10, color='red', linestyle=':', alpha=0.7)
    
    plt.suptitle('Heterogeneous Traffic: Time-Series Comparison', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/het_03_time_series.png", dpi=300)
    plt.close()
    print("  ✅ Saved: het_03_time_series.png")


# ═══════════════════════════════════════════════════════════════════
#  COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_comparative_analysis(df_homo, df_het):
    """Compare homogeneous vs heterogeneous performance."""
    print("Generating: Comparative analysis...")
    
    df_homo['t_dissolve_numeric'] = pd.to_numeric(df_homo['t_dissolve'], errors='coerce')
    df_het['t_dissolve_numeric'] = pd.to_numeric(df_het['t_dissolve'], errors='coerce')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution of dissolution times
    homo_times = df_homo['t_dissolve_numeric'].dropna()
    het_times = df_het['t_dissolve_numeric'].dropna()
    
    axes[0].hist(homo_times, bins=15, alpha=0.6, label='Homogeneous', color='blue', edgecolor='black')
    axes[0].hist(het_times, bins=15, alpha=0.6, label='Heterogeneous', color='red', edgecolor='black')
    axes[0].set_xlabel('Dissolution Time (s)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Dissolution Times', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot comparison
    data_to_plot = [homo_times, het_times]
    axes[1].boxplot(data_to_plot, labels=['Homogeneous', 'Heterogeneous'], 
                   patch_artist=True, showmeans=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2),
                   meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    axes[1].set_ylabel('Dissolution Time (s)', fontsize=12)
    axes[1].set_title('Statistical Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistical test
    t_stat, p_val = stats.ttest_ind(homo_times, het_times)
    axes[1].text(0.5, 0.95, f't-test: p={p_val:.4f}', 
                transform=axes[1].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Homogeneous vs Heterogeneous Traffic: Comparative Analysis', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comp_01_comparison.png", dpi=300)
    plt.close()
    print("  ✅ Saved: comp_01_comparison.png")


def generate_summary_statistics(df_homo, df_het):
    """Generate comprehensive summary statistics."""
    print("Generating: Summary statistics...")
    
    df_homo['t_dissolve_numeric'] = pd.to_numeric(df_homo['t_dissolve'], errors='coerce')
    df_het['t_dissolve_numeric'] = pd.to_numeric(df_het['t_dissolve'], errors='coerce')
    
    summary = {
        'Metric': [
            '=== HOMOGENEOUS TRAFFIC ===',
            'Total Scenarios',
            'Mean Dissolution Time (s)',
            'Median Dissolution Time (s)',
            'Std Dev (s)',
            'Min Dissolution Time (s)',
            'Max Dissolution Time (s)',
            'Perpetual Jams',
            '',
            '=== HETEROGENEOUS TRAFFIC ===',
            'Total Scenarios',
            'Mean Dissolution Time (s)',
            'Median Dissolution Time (s)',
            'Std Dev (s)',
            'Min Dissolution Time (s)',
            'Max Dissolution Time (s)',
            'Perpetual Jams',
            '',
            '=== STATISTICAL TEST ===',
            'T-statistic',
            'P-value',
            'Significant Difference',
        ],
        'Value': [
            '',
            f"{len(df_homo)}",
            f"{df_homo['t_dissolve_numeric'].mean():.2f}",
            f"{df_homo['t_dissolve_numeric'].median():.2f}",
            f"{df_homo['t_dissolve_numeric'].std():.2f}",
            f"{df_homo['t_dissolve_numeric'].min():.2f}",
            f"{df_homo['t_dissolve_numeric'].max():.2f}",
            f"{df_homo['jam_perpetuo'].sum()}",
            '',
            '',
            f"{len(df_het)}",
            f"{df_het['t_dissolve_numeric'].mean():.2f}",
            f"{df_het['t_dissolve_numeric'].median():.2f}",
            f"{df_het['t_dissolve_numeric'].std():.2f}",
            f"{df_het['t_dissolve_numeric'].min():.2f}",
            f"{df_het['t_dissolve_numeric'].max():.2f}",
            f"{df_het['jam_perpetuo'].sum()}",
            '',
            '',
            '',
            '',
            '',
        ]
    }
    
    # Statistical test
    homo_times = df_homo['t_dissolve_numeric'].dropna()
    het_times = df_het['t_dissolve_numeric'].dropna()
    t_stat, p_val = stats.ttest_ind(homo_times, het_times)
    
    summary['Value'][-3] = f"{t_stat:.4f}"
    summary['Value'][-2] = f"{p_val:.6f}"
    summary['Value'][-1] = 'Yes (p < 0.05)' if p_val < 0.05 else 'No (p >= 0.05)'
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv", index=False)
    print("  ✅ Saved: summary_statistics.csv")
    
    return summary_df


def find_best_worst_scenarios(df_homo, df_het):
    """Identify best and worst performing scenarios."""
    print("Generating: Best/worst scenarios...")
    
    df_homo['t_dissolve_numeric'] = pd.to_numeric(df_homo['t_dissolve'], errors='coerce')
    df_het['t_dissolve_numeric'] = pd.to_numeric(df_het['t_dissolve'], errors='coerce')
    
    # Remove perpetual jams
    df_homo_resolved = df_homo[df_homo['jam_perpetuo'] == 0]
    df_het_resolved = df_het[df_het['jam_perpetuo'] == 0]
    
    results = {
        'Category': [
            '=== HOMOGENEOUS: BEST ===',
            'Fastest Dissolution',
            'Lowest Max Stopped',
            '',
            '=== HOMOGENEOUS: WORST ===',
            'Slowest Dissolution',
            'Highest Max Stopped',
            '',
            '=== HETEROGENEOUS: BEST ===',
            'Fastest Dissolution',
            'Lowest Max Stopped',
            '',
            '=== HETEROGENEOUS: WORST ===',
            'Slowest Dissolution',
            'Highest Max Stopped',
        ],
        'Scenario': [
            '',
            df_homo_resolved.loc[df_homo_resolved['t_dissolve_numeric'].idxmin(), 'nombre'],
            df_homo.loc[df_homo['max_stopped'].idxmin(), 'nombre'],
            '',
            '',
            df_homo_resolved.loc[df_homo_resolved['t_dissolve_numeric'].idxmax(), 'nombre'],
            df_homo.loc[df_homo['max_stopped'].idxmax(), 'nombre'],
            '',
            '',
            df_het_resolved.loc[df_het_resolved['t_dissolve_numeric'].idxmin(), 'nombre'],
            df_het.loc[df_het['max_stopped'].idxmin(), 'nombre'],
            '',
            '',
            df_het_resolved.loc[df_het_resolved['t_dissolve_numeric'].idxmax(), 'nombre'],
            df_het.loc[df_het['max_stopped'].idxmax(), 'nombre'],
        ],
        'Value': [
            '',
            f"{df_homo_resolved['t_dissolve_numeric'].min():.1f}s",
            f"{df_homo['max_stopped'].min():.0f} vehicles",
            '',
            '',
            f"{df_homo_resolved['t_dissolve_numeric'].max():.1f}s",
            f"{df_homo['max_stopped'].max():.0f} vehicles",
            '',
            '',
            f"{df_het_resolved['t_dissolve_numeric'].min():.1f}s",
            f"{df_het['max_stopped'].min():.0f} vehicles",
            '',
            '',
            f"{df_het_resolved['t_dissolve_numeric'].max():.1f}s",
            f"{df_het['max_stopped'].max():.0f} vehicles",
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/best_worst_scenarios.csv", index=False)
    print("  ✅ Saved: best_worst_scenarios.csv")
    
    return results_df


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE TRAFFIC ANALYSIS")
    print("=" * 70 + "\n")
    
    # Load data
    df_homo = load_data("resultados_homogeneo_master.csv")
    df_het = load_data("resultados_heterogeneo_final_master.csv")
    
    if df_homo is None and df_het is None:
        print("\n❌ No data files found. Run simulations first!")
        return
    
    print(f"\n📊 Generating visualizations...\n")
    
    # Homogeneous analysis
    if df_homo is not None:
        print("=== HOMOGENEOUS ANALYSIS ===")
        plot_homogeneous_heatmap(df_homo)
        plot_homogeneous_trends(df_homo)
        plot_homogeneous_time_series(df_homo)
    
    # Heterogeneous analysis
    if df_het is not None:
        print("\n=== HETEROGENEOUS ANALYSIS ===")
        plot_heterogeneous_heatmap(df_het)
        plot_heterogeneous_trends(df_het)
        plot_heterogeneous_time_series(df_het)
    
    # Comparative analysis
    if df_homo is not None and df_het is not None:
        print("\n=== COMPARATIVE ANALYSIS ===")
        plot_comparative_analysis(df_homo, df_het)
        generate_summary_statistics(df_homo, df_het)
        find_best_worst_scenarios(df_homo, df_het)
    
    print("\n" + "=" * 70)
    print(f"✅ Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("=" * 70 + "\n")
    
    # Print file summary
    print("Generated files:")
    files = sorted(Path(OUTPUT_DIR).glob("*"))
    for f in files:
        print(f"  • {f.name}")
    print()


if __name__ == "__main__":
    main()
