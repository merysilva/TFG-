"""
analyze_heterogeneous_traffic.py — Analysis and Visualization
==============================================================
Generates comprehensive graphs and data analysis from the
heterogeneous traffic flow simulation results.

Usage:
    python analyze_heterogeneous_traffic.py

Requirements:
    - resultados_heterogeneo/ folder with individual scenario CSVs
    - resultados_heterogeneo_master.csv with summary statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_master_data():
    """Load the master summary file."""
    try:
        df = pd.read_csv("resultados_heterogeneo_master.csv")
        print(f"✅ Loaded master data: {len(df)} scenarios")
        return df
    except FileNotFoundError:
        print("❌ Error: resultados_heterogeneo_master.csv not found")
        return None


def load_scenario_data(scenario_name):
    """Load time-series data for a specific scenario."""
    filepath = f"resultados_heterogeneo/{scenario_name}_datos.csv"
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"⚠️  Warning: {filepath} not found")
        return None


# ═══════════════════════════════════════════════════════════════════
#  MASTER DATA VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def plot_dissolution_time_heatmap(df):
    """Heatmap: Traffic jam dissolution time by truck % and aggression %."""
    print("Generating: Dissolution time heatmap...")
    
    # Convert t_dissolve to numeric (NaN for perpetual jams)
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    # Create pivot table
    pivot = df.pivot_table(
        values='t_dissolve_numeric',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    
    # Convert to percentages for labels
    pivot.index = (pivot.index * 100).astype(int)
    pivot.columns = (pivot.columns * 100).astype(int)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Dissolution Time (seconds)'},
                linewidths=0.5)
    plt.xlabel('Aggressive Drivers (%)', fontsize=12)
    plt.ylabel('Trucks (%)', fontsize=12)
    plt.title('Traffic Jam Dissolution Time\n(Lower is better)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_dissolution_time_heatmap.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 01_dissolution_time_heatmap.png")


def plot_throughput_heatmap(df):
    """Heatmap: Vehicle throughput by truck % and aggression %."""
    print("Generating: Throughput heatmap...")
    
    pivot = df.pivot_table(
        values='final_throughput',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    
    pivot.index = (pivot.index * 100).astype(int)
    pivot.columns = (pivot.columns * 100).astype(int)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', 
                cbar_kws={'label': 'Total Vehicles Passed'},
                linewidths=0.5)
    plt.xlabel('Aggressive Drivers (%)', fontsize=12)
    plt.ylabel('Trucks (%)', fontsize=12)
    plt.title('Total Throughput Performance\n(Higher is better)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_throughput_heatmap.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 02_throughput_heatmap.png")


def plot_max_stopped_heatmap(df):
    """Heatmap: Maximum stopped vehicles by truck % and aggression %."""
    print("Generating: Maximum stopped vehicles heatmap...")
    
    pivot = df.pivot_table(
        values='max_stopped',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    
    pivot.index = (pivot.index * 100).astype(int)
    pivot.columns = (pivot.columns * 100).astype(int)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Reds', 
                cbar_kws={'label': 'Maximum Stopped Vehicles'},
                linewidths=0.5)
    plt.xlabel('Aggressive Drivers (%)', fontsize=12)
    plt.ylabel('Trucks (%)', fontsize=12)
    plt.title('Maximum Congestion Level\n(Lower is better)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_max_stopped_heatmap.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 03_max_stopped_heatmap.png")


def plot_safety_metrics(df):
    """Bar charts: Safety metrics (near collisions, hard brakes)."""
    print("Generating: Safety metrics comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Near collisions
    pivot_nc = df.pivot_table(
        values='near_collisions',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    pivot_nc.index = (pivot_nc.index * 100).astype(int)
    pivot_nc.columns = (pivot_nc.columns * 100).astype(int)
    
    sns.heatmap(pivot_nc, annot=True, fmt='.0f', cmap='OrRd', ax=axes[0],
                cbar_kws={'label': 'Near Collisions'}, linewidths=0.5)
    axes[0].set_xlabel('Aggressive Drivers (%)', fontsize=11)
    axes[0].set_ylabel('Trucks (%)', fontsize=11)
    axes[0].set_title('Near Collisions (gap < 2m at v > 2m/s)', fontsize=12, fontweight='bold')
    
    # Hard brakes
    pivot_hb = df.pivot_table(
        values='hard_brakes',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    pivot_hb.index = (pivot_hb.index * 100).astype(int)
    pivot_hb.columns = (pivot_hb.columns * 100).astype(int)
    
    sns.heatmap(pivot_hb, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'Hard Braking Events'}, linewidths=0.5)
    axes[1].set_xlabel('Aggressive Drivers (%)', fontsize=11)
    axes[1].set_ylabel('Trucks (%)', fontsize=11)
    axes[1].set_title('Hard Braking Events (a < -4 m/s²)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_safety_metrics.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 04_safety_metrics.png")


def plot_truck_effect(df):
    """Line plot: Effect of truck percentage on key metrics."""
    print("Generating: Truck percentage effect...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('t_dissolve_numeric', 'Dissolution Time (s)', axes[0, 0]),
        ('final_throughput', 'Total Throughput (vehicles)', axes[0, 1]),
        ('max_stopped', 'Maximum Stopped Vehicles', axes[1, 0]),
        ('near_collisions', 'Near Collisions', axes[1, 1])
    ]
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    df['truck_pct_label'] = (df['truck_pct'] * 100).astype(int)
    df['aggressive_pct_label'] = (df['aggressive_pct'] * 100).astype(int)
    
    for metric, ylabel, ax in metrics:
        for aggr in sorted(df['aggressive_pct'].unique()):
            subset = df[df['aggressive_pct'] == aggr]
            aggr_label = int(aggr * 100)
            ax.plot(subset['truck_pct_label'], subset[metric], 
                   marker='o', linewidth=2, label=f'{aggr_label}% Aggressive')
        
        ax.set_xlabel('Truck Percentage (%)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} vs Truck %', fontsize=12, fontweight='bold')
        ax.legend(title='Driver Aggression', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_truck_effect.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 05_truck_effect.png")


def plot_aggression_effect(df):
    """Line plot: Effect of driver aggression on key metrics."""
    print("Generating: Driver aggression effect...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('t_dissolve_numeric', 'Dissolution Time (s)', axes[0, 0]),
        ('final_throughput', 'Total Throughput (vehicles)', axes[0, 1]),
        ('max_stopped', 'Maximum Stopped Vehicles', axes[1, 0]),
        ('hard_brakes', 'Hard Braking Events', axes[1, 1])
    ]
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    df['truck_pct_label'] = (df['truck_pct'] * 100).astype(int)
    df['aggressive_pct_label'] = (df['aggressive_pct'] * 100).astype(int)
    
    for metric, ylabel, ax in metrics:
        for truck in sorted(df['truck_pct'].unique()):
            subset = df[df['truck_pct'] == truck]
            truck_label = int(truck * 100)
            ax.plot(subset['aggressive_pct_label'], subset[metric], 
                   marker='s', linewidth=2, label=f'{truck_label}% Trucks')
        
        ax.set_xlabel('Aggressive Driver Percentage (%)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} vs Aggression %', fontsize=12, fontweight='bold')
        ax.legend(title='Fleet Composition', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_aggression_effect.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 06_aggression_effect.png")


def plot_perpetual_jams(df):
    """Bar chart: Perpetual jam occurrences."""
    print("Generating: Perpetual jam analysis...")
    
    pivot = df.pivot_table(
        values='jam_perpetuo',
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='sum'
    )
    
    pivot.index = (pivot.index * 100).astype(int)
    pivot.columns = (pivot.columns * 100).astype(int)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Perpetual Jam (1=Yes, 0=No)'},
                linewidths=0.5, vmin=0, vmax=1)
    plt.xlabel('Aggressive Drivers (%)', fontsize=12)
    plt.ylabel('Trucks (%)', fontsize=12)
    plt.title('Perpetual Traffic Jam Occurrence\n(Red indicates jam did not dissolve)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_perpetual_jams.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 07_perpetual_jams.png")


# ═══════════════════════════════════════════════════════════════════
#  TIME-SERIES VISUALIZATIONS (SELECTED SCENARIOS)
# ═══════════════════════════════════════════════════════════════════

def plot_velocity_evolution(scenarios_to_plot):
    """Time-series: Average velocity evolution for selected scenarios."""
    print("Generating: Velocity evolution time-series...")
    
    plt.figure(figsize=(16, 8))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            plt.plot(df['segundo'], df['avg_velocity'], linewidth=2, label=scenario_name)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Average Velocity (m/s)', fontsize=12)
    plt.title('Average Velocity Evolution Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Disturbance Start')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_velocity_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 08_velocity_evolution.png")


def plot_stopped_vehicles_evolution(scenarios_to_plot):
    """Time-series: Stopped vehicles evolution for selected scenarios."""
    print("Generating: Stopped vehicles evolution...")
    
    plt.figure(figsize=(16, 8))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            plt.plot(df['segundo'], df['vehicles_stopped'], linewidth=2, label=scenario_name)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of Stopped Vehicles', fontsize=12)
    plt.title('Stopped Vehicles Evolution Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=7, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_stopped_vehicles_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 09_stopped_vehicles_evolution.png")


def plot_throughput_evolution(scenarios_to_plot):
    """Time-series: Cumulative throughput for selected scenarios."""
    print("Generating: Throughput evolution...")
    
    plt.figure(figsize=(16, 8))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            plt.plot(df['segundo'], df['cumulative_throughput'], linewidth=2, label=scenario_name)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Cumulative Throughput (vehicles)', fontsize=12)
    plt.title('Cumulative Throughput Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_throughput_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 10_throughput_evolution.png")


def plot_detailed_scenario_analysis(scenario_name):
    """Comprehensive multi-panel analysis for a single scenario."""
    print(f"Generating: Detailed analysis for {scenario_name}...")
    
    df = load_scenario_data(scenario_name)
    if df is None:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    
    # Average velocity
    axes[0, 0].plot(df['segundo'], df['avg_velocity'], linewidth=2, color='blue')
    axes[0, 0].fill_between(df['segundo'], df['min_velocity'], df['max_velocity'], alpha=0.3)
    axes[0, 0].set_xlabel('Time (s)', fontsize=10)
    axes[0, 0].set_ylabel('Velocity (m/s)', fontsize=10)
    axes[0, 0].set_title('Velocity Range (avg ± min/max)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=7, color='red', linestyle='--', alpha=0.5)
    
    # Vehicle states
    axes[0, 1].plot(df['segundo'], df['vehicles_cruising'], label='Cruising', linewidth=2)
    axes[0, 1].plot(df['segundo'], df['vehicles_braking'], label='Braking', linewidth=2)
    axes[0, 1].plot(df['segundo'], df['vehicles_stopped'], label='Stopped', linewidth=2)
    axes[0, 1].plot(df['segundo'], df['vehicles_accelerating'], label='Accelerating', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)', fontsize=10)
    axes[0, 1].set_ylabel('Number of Vehicles', fontsize=10)
    axes[0, 1].set_title('Vehicle States Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=7, color='red', linestyle='--', alpha=0.5)
    
    # Gap analysis
    axes[1, 0].plot(df['segundo'], df['avg_gap'], label='Average Gap', linewidth=2)
    axes[1, 0].plot(df['segundo'], df['min_gap'], label='Minimum Gap', linewidth=2, color='red')
    axes[1, 0].axhline(y=2.0, color='orange', linestyle='--', linewidth=1, label='Danger Threshold')
    axes[1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[1, 0].set_ylabel('Gap (m)', fontsize=10)
    axes[1, 0].set_title('Following Distance', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=7, color='red', linestyle='--', alpha=0.5)
    
    # Throughput
    axes[1, 1].plot(df['segundo'], df['throughput_per_min'], linewidth=2, color='green')
    axes[1, 1].set_xlabel('Time (s)', fontsize=10)
    axes[1, 1].set_ylabel('Throughput (vehicles/min)', fontsize=10)
    axes[1, 1].set_title('Instantaneous Throughput Rate', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=7, color='red', linestyle='--', alpha=0.5)
    
    # Acceleration distribution
    axes[2, 0].plot(df['segundo'], df['avg_acceleration'], linewidth=2, label='Average')
    axes[2, 0].fill_between(df['segundo'], df['min_acceleration'], df['max_acceleration'], alpha=0.3)
    axes[2, 0].axhline(y=-4.0, color='red', linestyle='--', linewidth=1, label='Hard Brake Threshold')
    axes[2, 0].set_xlabel('Time (s)', fontsize=10)
    axes[2, 0].set_ylabel('Acceleration (m/s²)', fontsize=10)
    axes[2, 0].set_title('Acceleration Range', fontsize=11, fontweight='bold')
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axvline(x=7, color='red', linestyle='--', alpha=0.5)
    
    # Cumulative safety metrics
    axes[2, 1].plot(df['segundo'], df['near_collisions'], label='Near Collisions', linewidth=2)
    axes[2, 1].plot(df['segundo'], df['hard_brakes'], label='Hard Brakes', linewidth=2)
    axes[2, 1].set_xlabel('Time (s)', fontsize=10)
    axes[2, 1].set_ylabel('Cumulative Count', fontsize=10)
    axes[2, 1].set_title('Cumulative Safety Events', fontsize=11, fontweight='bold')
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axvline(x=7, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Detailed Analysis: {scenario_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_detailed_{scenario_name}.png", dpi=300)
    plt.close()
    print(f"  ✅ Saved: 11_detailed_{scenario_name}.png")


# ═══════════════════════════════════════════════════════════════════
#  DATA EXTRACTION AND SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics table."""
    print("Generating: Summary statistics...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    summary = {
        'Metric': [
            'Mean Dissolution Time (s)',
            'Median Dissolution Time (s)',
            'Std Deviation Dissolution Time',
            'Min Dissolution Time (s)',
            'Max Dissolution Time (s)',
            '',
            'Mean Throughput (vehicles)',
            'Median Throughput (vehicles)',
            'Min Throughput (vehicles)',
            'Max Throughput (vehicles)',
            '',
            'Mean Max Stopped Vehicles',
            'Max Max Stopped Vehicles',
            '',
            'Total Near Collisions',
            'Total Hard Brakes',
            'Perpetual Jams Count',
        ],
        'Value': [
            f"{df['t_dissolve_numeric'].mean():.2f}",
            f"{df['t_dissolve_numeric'].median():.2f}",
            f"{df['t_dissolve_numeric'].std():.2f}",
            f"{df['t_dissolve_numeric'].min():.2f}",
            f"{df['t_dissolve_numeric'].max():.2f}",
            '',
            f"{df['final_throughput'].mean():.1f}",
            f"{df['final_throughput'].median():.1f}",
            f"{df['final_throughput'].min():.0f}",
            f"{df['final_throughput'].max():.0f}",
            '',
            f"{df['max_stopped'].mean():.2f}",
            f"{df['max_stopped'].max():.0f}",
            '',
            f"{df['near_collisions'].sum():.0f}",
            f"{df['hard_brakes'].sum():.0f}",
            f"{df['jam_perpetuo'].sum():.0f}",
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv", index=False)
    print("  ✅ Saved: summary_statistics.csv")
    
    return summary_df


def find_best_worst_scenarios(df):
    """Identify best and worst performing scenarios."""
    print("Generating: Best/worst scenarios analysis...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    # Best scenarios (fastest dissolution, highest throughput, fewest safety issues)
    best_dissolution = df.loc[df['t_dissolve_numeric'].idxmin()]
    best_throughput = df.loc[df['final_throughput'].idxmax()]
    safest = df.loc[(df['near_collisions'] + df['hard_brakes']).idxmin()]
    
    # Worst scenarios
    worst_dissolution = df.loc[df['t_dissolve_numeric'].idxmax()]
    worst_throughput = df.loc[df['final_throughput'].idxmin()]
    most_dangerous = df.loc[(df['near_collisions'] + df['hard_brakes']).idxmax()]
    
    results = {
        'Category': [
            'Fastest Jam Dissolution',
            'Highest Throughput',
            'Safest (Fewest Safety Events)',
            '',
            'Slowest Jam Dissolution',
            'Lowest Throughput',
            'Most Dangerous (Most Safety Events)',
        ],
        'Scenario': [
            best_dissolution['nombre'],
            best_throughput['nombre'],
            safest['nombre'],
            '',
            worst_dissolution['nombre'],
            worst_throughput['nombre'],
            most_dangerous['nombre'],
        ],
        'Value': [
            f"{best_dissolution['t_dissolve_numeric']:.1f}s",
            f"{best_throughput['final_throughput']:.0f} vehicles",
            f"{safest['near_collisions'] + safest['hard_brakes']:.0f} events",
            '',
            f"{worst_dissolution['t_dissolve_numeric']:.1f}s",
            f"{worst_throughput['final_throughput']:.0f} vehicles",
            f"{most_dangerous['near_collisions'] + most_dangerous['hard_brakes']:.0f} events",
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/best_worst_scenarios.csv", index=False)
    print("  ✅ Saved: best_worst_scenarios.csv")
    
    return results_df


def generate_correlation_analysis(df):
    """Generate correlation matrix for key metrics."""
    print("Generating: Correlation analysis...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    corr_cols = ['truck_pct', 'aggressive_pct', 't_dissolve_numeric', 
                 'final_throughput', 'max_stopped', 'near_collisions', 'hard_brakes']
    
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_correlation_matrix.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 12_correlation_matrix.png")
    
    corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC ANALYSIS")
    print("=" * 70 + "\n")
    
    # Load master data
    master_df = load_master_data()
    if master_df is None:
        print("\n❌ Cannot proceed without master data file.")
        return
    
    print(f"\n📊 Generating visualizations...\n")
    
    # Master data visualizations
    plot_dissolution_time_heatmap(master_df)
    plot_throughput_heatmap(master_df)
    plot_max_stopped_heatmap(master_df)
    plot_safety_metrics(master_df)
    plot_truck_effect(master_df)
    plot_aggression_effect(master_df)
    plot_perpetual_jams(master_df)
    
    # Time-series visualizations for selected scenarios
    scenarios_to_compare = [
        'Exp_T00_A000',  # Baseline: no trucks, all cautious
        'Exp_T00_A100',  # No trucks, all aggressive
        'Exp_T50_A000',  # 50% trucks, all cautious
        'Exp_T50_A100',  # 50% trucks, all aggressive
        'Exp_T75_A050',  # 75% trucks, mixed drivers
    ]
    
    plot_velocity_evolution(scenarios_to_compare)
    plot_stopped_vehicles_evolution(scenarios_to_compare)
    plot_throughput_evolution(scenarios_to_compare)
    
    # Detailed analysis for extreme scenarios
    plot_detailed_scenario_analysis('Exp_T00_A000')
    plot_detailed_scenario_analysis('Exp_T75_A100')
    
    # Statistical analysis
    print(f"\n📈 Generating statistical summaries...\n")
    generate_summary_statistics(master_df)
    find_best_worst_scenarios(master_df)
    generate_correlation_analysis(master_df)
    
    print("\n" + "=" * 70)
    print(f"✅ Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("=" * 70 + "\n")
    
    # Print summary
    print("Generated files:")
    files = sorted(Path(OUTPUT_DIR).glob("*"))
    for f in files:
        print(f"  • {f.name}")
    print()


if __name__ == "__main__":
    main()
