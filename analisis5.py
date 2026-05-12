"""
analyze_heterogeneous_v2.py — Comprehensive Traffic Analysis
=============================================================
Generates extensive graphs and statistical analysis from the
improved heterogeneous traffic simulation results.

Includes:
    - All metrics: dissolution time, throughput, safety, speed variance,
      gap distributions, wave propagation
    - Vehicle type comparisons (cars vs trucks)
    - Driver behavior analysis (aggressive vs cautious)
    - Time-series evolution plots
    - Statistical correlations and significance tests

Usage:
    python analyze_heterogeneous_v2.py
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

OUTPUT_DIR = "analysis_results_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_master_data():
    """Load the master summary file."""
    try:
        df = pd.read_csv("resultados_heterogeneo_v2_master.csv")
        print(f"✅ Loaded master data: {len(df)} scenarios")
        return df
    except FileNotFoundError:
        print("❌ Error: resultados_heterogeneo_v2_master.csv not found")
        print("   Run simulador_heterogeneo_v2.py first!")
        return None


def load_scenario_data(scenario_name):
    """Load time-series data for a specific scenario."""
    filepath = f"resultados_heterogeneo_v2/{scenario_name}_datos.csv"
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"⚠️  Warning: {filepath} not found")
        return None


# ═══════════════════════════════════════════════════════════════════
#  MASTER DATA HEATMAPS
# ═══════════════════════════════════════════════════════════════════

def create_heatmap(df, value_col, title, filename, cmap='RdYlGn_r', fmt='.1f'):
    """Generic heatmap creator."""
    print(f"Generating: {title}...")
    
    # Convert to numeric if needed
    if df[value_col].dtype == 'object':
        df[f'{value_col}_numeric'] = pd.to_numeric(df[value_col], errors='coerce')
        value_col = f'{value_col}_numeric'
    
    pivot = df.pivot_table(
        values=value_col,
        index='truck_pct',
        columns='aggressive_pct',
        aggfunc='mean'
    )
    
    pivot.index = (pivot.index * 100).astype(int)
    pivot.columns = (pivot.columns * 100).astype(int)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, 
                cbar_kws={'label': title.split('\n')[0]},
                linewidths=0.5)
    plt.xlabel('Aggressive Drivers (%)', fontsize=12)
    plt.ylabel('Trucks (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    plt.close()
    print(f"  ✅ Saved: {filename}")


def plot_all_heatmaps(df):
    """Generate all heatmap visualizations."""
    heatmaps = [
        ('t_dissolve', 'Traffic Jam Dissolution Time\n(Lower is better)', 
         '01_dissolution_time_heatmap.png', 'RdYlGn_r', '.1f'),
        ('final_throughput', 'Total Throughput Performance\n(Higher is better)', 
         '02_throughput_heatmap.png', 'YlGnBu', '.0f'),
        ('max_stopped', 'Maximum Congestion Level\n(Lower is better)', 
         '03_max_stopped_heatmap.png', 'Reds', '.0f'),
        ('near_collisions', 'Near Collisions\n(Lower is better)', 
         '04_near_collisions_heatmap.png', 'OrRd', '.0f'),
        ('hard_brakes', 'Hard Braking Events\n(Lower is better)', 
         '05_hard_brakes_heatmap.png', 'YlOrRd', '.0f'),
        ('wave_speed', 'Jam Wave Propagation Speed (m/s)\n(Lower is better)', 
         '06_wave_speed_heatmap.png', 'RdYlGn_r', '.2f'),
    ]
    
    for value_col, title, filename, cmap, fmt in heatmaps:
        create_heatmap(df, value_col, title, filename, cmap, fmt)


# ═══════════════════════════════════════════════════════════════════
#  TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_truck_effect(df):
    """Effect of truck percentage on all key metrics."""
    print("Generating: Truck percentage effect...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    df['wave_speed_numeric'] = pd.to_numeric(df['wave_speed'], errors='coerce')
    df['truck_pct_label'] = (df['truck_pct'] * 100).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = [
        ('t_dissolve_numeric', 'Dissolution Time (s)', axes[0, 0]),
        ('final_throughput', 'Total Throughput (vehicles)', axes[0, 1]),
        ('max_stopped', 'Maximum Stopped Vehicles', axes[0, 2]),
        ('wave_speed_numeric', 'Wave Propagation Speed (m/s)', axes[1, 0]),
        ('near_collisions', 'Near Collisions', axes[1, 1]),
        ('hard_brakes', 'Hard Braking Events', axes[1, 2])
    ]
    
    for metric, ylabel, ax in metrics:
        for aggr in sorted(df['aggressive_pct'].unique()):
            subset = df[df['aggressive_pct'] == aggr]
            aggr_label = int(aggr * 100)
            ax.plot(subset['truck_pct_label'], subset[metric], 
                   marker='o', linewidth=2, markersize=8,
                   label=f'{aggr_label}% Aggressive')
        
        ax.set_xlabel('Truck Percentage (%)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(title='Driver Aggression', fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Truck Percentage on System Performance', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_truck_effect.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 07_truck_effect.png")


def plot_aggression_effect(df):
    """Effect of driver aggression on all key metrics."""
    print("Generating: Driver aggression effect...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    df['wave_speed_numeric'] = pd.to_numeric(df['wave_speed'], errors='coerce')
    df['aggressive_pct_label'] = (df['aggressive_pct'] * 100).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = [
        ('t_dissolve_numeric', 'Dissolution Time (s)', axes[0, 0]),
        ('final_throughput', 'Total Throughput (vehicles)', axes[0, 1]),
        ('max_stopped', 'Maximum Stopped Vehicles', axes[0, 2]),
        ('wave_speed_numeric', 'Wave Propagation Speed (m/s)', axes[1, 0]),
        ('near_collisions', 'Near Collisions', axes[1, 1]),
        ('hard_brakes', 'Hard Braking Events', axes[1, 2])
    ]
    
    for metric, ylabel, ax in metrics:
        for truck in sorted(df['truck_pct'].unique()):
            subset = df[df['truck_pct'] == truck]
            truck_label = int(truck * 100)
            ax.plot(subset['aggressive_pct_label'], subset[metric], 
                   marker='s', linewidth=2, markersize=8,
                   label=f'{truck_label}% Trucks')
        
        ax.set_xlabel('Aggressive Driver Percentage (%)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(title='Fleet Composition', fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Driver Aggression on System Performance', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_aggression_effect.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 08_aggression_effect.png")


# ═══════════════════════════════════════════════════════════════════
#  TIME-SERIES ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_velocity_evolution(scenarios_to_plot):
    """Time-series: Average velocity evolution."""
    print("Generating: Velocity evolution time-series...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            # Overall average
            axes[0].plot(df['segundo'], df['avg_velocity'], 
                        linewidth=2, label=scenario_name, alpha=0.8)
            
            # Cars vs Trucks
            axes[1].plot(df['segundo'], df['car_mean_velocity'], 
                        linewidth=2, linestyle='-', label=f'{scenario_name} (Cars)')
            axes[1].plot(df['segundo'], df['truck_mean_velocity'], 
                        linewidth=2, linestyle='--', label=f'{scenario_name} (Trucks)')
    
    for ax in axes:
        ax.axvline(x=10, color='red', linestyle=':', linewidth=1.5, 
                  alpha=0.7, label='Disturbance Start')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    axes[0].set_ylabel('Average Velocity (m/s)', fontsize=12)
    axes[0].set_title('System-Wide Average Velocity', fontsize=13, fontweight='bold')
    
    axes[1].set_ylabel('Velocity by Type (m/s)', fontsize=12)
    axes[1].set_title('Velocity Comparison: Cars vs Trucks', fontsize=13, fontweight='bold')
    
    plt.suptitle('Velocity Evolution Over Time', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_velocity_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 09_velocity_evolution.png")


def plot_speed_variance_evolution(scenarios_to_plot):
    """Time-series: Speed variance by vehicle type."""
    print("Generating: Speed variance evolution...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            axes[0].plot(df['segundo'], df['car_velocity_variance'], 
                        linewidth=2, label=scenario_name)
            axes[1].plot(df['segundo'], df['truck_velocity_variance'], 
                        linewidth=2, label=scenario_name)
    
    for ax in axes:
        ax.axvline(x=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    axes[0].set_ylabel('Variance (m²/s²)', fontsize=12)
    axes[0].set_title('Car Speed Variance', fontsize=13, fontweight='bold')
    
    axes[1].set_ylabel('Variance (m²/s²)', fontsize=12)
    axes[1].set_title('Truck Speed Variance', fontsize=13, fontweight='bold')
    
    plt.suptitle('Speed Variance by Vehicle Type', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_speed_variance_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 10_speed_variance_evolution.png")


def plot_gap_evolution(scenarios_to_plot):
    """Time-series: Gap distribution evolution."""
    print("Generating: Gap distribution evolution...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            # Mean gap
            axes[0].plot(df['segundo'], df['avg_gap'], 
                        linewidth=2, label=scenario_name)
            
            # Min gap (safety critical)
            axes[1].plot(df['segundo'], df['min_gap'], 
                        linewidth=2, label=scenario_name)
    
    axes[0].set_ylabel('Average Gap (m)', fontsize=12)
    axes[0].set_title('Mean Following Distance', fontsize=13, fontweight='bold')
    
    axes[1].set_ylabel('Minimum Gap (m)', fontsize=12)
    axes[1].set_title('Minimum Following Distance (Safety Critical)', 
                     fontsize=13, fontweight='bold')
    axes[1].axhline(y=2.0, color='orange', linestyle='--', linewidth=2, 
                   label='Danger Threshold (2m)')
    
    for ax in axes:
        ax.axvline(x=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.suptitle('Gap Distribution Evolution', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_gap_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 11_gap_evolution.png")


def plot_vehicle_states_evolution(scenarios_to_plot):
    """Time-series: Vehicle state distribution."""
    print("Generating: Vehicle states evolution...")
    
    n_scenarios = len(scenarios_to_plot)
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(16, 5 * n_scenarios))
    
    if n_scenarios == 1:
        axes = [axes]
    
    for ax, scenario_name in zip(axes, scenarios_to_plot):
        df = load_scenario_data(scenario_name)
        if df is not None:
            ax.plot(df['segundo'], df['vehicles_cruising'], 
                   label='Cruising', linewidth=2)
            ax.plot(df['segundo'], df['vehicles_braking'], 
                   label='Braking', linewidth=2)
            ax.plot(df['segundo'], df['vehicles_stopped'], 
                   label='Stopped', linewidth=2)
            ax.plot(df['segundo'], df['vehicles_accelerating'], 
                   label='Accelerating', linewidth=2)
            
            ax.axvline(x=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time (seconds)', fontsize=11)
            ax.set_ylabel('Number of Vehicles', fontsize=11)
            ax.set_title(f'{scenario_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Vehicle State Distribution Over Time', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_vehicle_states_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 12_vehicle_states_evolution.png")


def plot_throughput_evolution(scenarios_to_plot):
    """Time-series: Cumulative throughput comparison."""
    print("Generating: Throughput evolution...")
    
    plt.figure(figsize=(16, 8))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            plt.plot(df['segundo'], df['cumulative_throughput'], 
                    linewidth=2.5, label=scenario_name, alpha=0.8)
    
    plt.axvline(x=10, color='red', linestyle=':', linewidth=1.5, 
               alpha=0.7, label='Disturbance Start')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Cumulative Throughput (vehicles)', fontsize=12)
    plt.title('Cumulative Throughput Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/13_throughput_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 13_throughput_evolution.png")


def plot_safety_events_evolution(scenarios_to_plot):
    """Time-series: Cumulative safety events."""
    print("Generating: Safety events evolution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for scenario_name in scenarios_to_plot:
        df = load_scenario_data(scenario_name)
        if df is not None:
            axes[0].plot(df['segundo'], df['near_collisions'], 
                        linewidth=2, label=scenario_name)
            axes[1].plot(df['segundo'], df['hard_brakes'], 
                        linewidth=2, label=scenario_name)
    
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Cumulative Near Collisions', fontsize=12)
    axes[0].set_title('Near Collision Events (gap < 2m @ v > 2m/s)', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Cumulative Hard Brakes', fontsize=12)
    axes[1].set_title('Hard Braking Events (a < -4 m/s²)', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.suptitle('Safety Event Evolution', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/14_safety_events_evolution.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 14_safety_events_evolution.png")


# ═══════════════════════════════════════════════════════════════════
#  DETAILED SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_detailed_scenario(scenario_name):
    """Comprehensive 6-panel analysis for a single scenario."""
    print(f"Generating: Detailed analysis for {scenario_name}...")
    
    df = load_scenario_data(scenario_name)
    if df is None:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    
    # 1. Velocity with cars vs trucks
    ax = axes[0, 0]
    ax.plot(df['segundo'], df['avg_velocity'], linewidth=2.5, 
           label='System Average', color='black')
    ax.plot(df['segundo'], df['car_mean_velocity'], linewidth=2, 
           label='Cars', linestyle='-', alpha=0.7)
    ax.plot(df['segundo'], df['truck_mean_velocity'], linewidth=2, 
           label='Trucks', linestyle='--', alpha=0.7)
    ax.fill_between(df['segundo'], df['min_velocity'], df['max_velocity'], 
                    alpha=0.2, label='Min-Max Range')
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Velocity (m/s)', fontsize=10)
    ax.set_title('Velocity Profiles', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Vehicle states
    ax = axes[0, 1]
    ax.plot(df['segundo'], df['vehicles_cruising'], label='Cruising', linewidth=2)
    ax.plot(df['segundo'], df['vehicles_braking'], label='Braking', linewidth=2)
    ax.plot(df['segundo'], df['vehicles_stopped'], label='Stopped', linewidth=2)
    ax.plot(df['segundo'], df['vehicles_accelerating'], label='Accelerating', linewidth=2)
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Number of Vehicles', fontsize=10)
    ax.set_title('Vehicle State Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Gap analysis
    ax = axes[1, 0]
    ax.plot(df['segundo'], df['avg_gap'], label='Average Gap', linewidth=2)
    ax.plot(df['segundo'], df['median_gap'], label='Median Gap', linewidth=2, linestyle='--')
    ax.plot(df['segundo'], df['min_gap'], label='Minimum Gap', linewidth=2, color='red')
    ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, label='Danger Threshold')
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Gap (m)', fontsize=10)
    ax.set_title('Following Distance Analysis', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Speed variance by type
    ax = axes[1, 1]
    ax.plot(df['segundo'], df['car_velocity_variance'], 
           label='Car Variance', linewidth=2)
    ax.plot(df['segundo'], df['truck_velocity_variance'], 
           label='Truck Variance', linewidth=2)
    ax.plot(df['segundo'], df['velocity_variance'], 
           label='System Variance', linewidth=2, color='black', alpha=0.5)
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Variance (m²/s²)', fontsize=10)
    ax.set_title('Speed Variance by Vehicle Type', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. Throughput
    ax = axes[2, 0]
    ax.plot(df['segundo'], df['cumulative_throughput'], linewidth=2.5, color='green')
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Cumulative Throughput (vehicles)', fontsize=10)
    ax.set_title('Cumulative Throughput', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Safety metrics
    ax = axes[2, 1]
    ax2 = ax.twinx()
    line1 = ax.plot(df['segundo'], df['near_collisions'], 
                    label='Near Collisions', linewidth=2, color='orange')
    line2 = ax2.plot(df['segundo'], df['hard_brakes'], 
                     label='Hard Brakes', linewidth=2, color='red')
    ax.axvline(x=10, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Cumulative Near Collisions', fontsize=10, color='orange')
    ax2.set_ylabel('Cumulative Hard Brakes', fontsize=10, color='red')
    ax.set_title('Cumulative Safety Events', fontsize=11, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Detailed Analysis: {scenario_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/15_detailed_{scenario_name}.png", dpi=300)
    plt.close()
    print(f"  ✅ Saved: 15_detailed_{scenario_name}.png")


# ═══════════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics."""
    print("Generating: Summary statistics...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    df['wave_speed_numeric'] = pd.to_numeric(df['wave_speed'], errors='coerce')
    
    summary = {
        'Metric': [
            '=== DISSOLUTION TIME ===',
            'Mean (s)',
            'Median (s)',
            'Std Dev (s)',
            'Min (s)',
            'Max (s)',
            '',
            '=== THROUGHPUT ===',
            'Mean (vehicles)',
            'Median (vehicles)',
            'Std Dev (vehicles)',
            'Min (vehicles)',
            'Max (vehicles)',
            '',
            '=== WAVE PROPAGATION ===',
            'Mean Speed (m/s)',
            'Median Speed (m/s)',
            'Std Dev (m/s)',
            '',
            '=== CONGESTION ===',
            'Mean Max Stopped',
            'Max Max Stopped',
            '',
            '=== SAFETY ===',
            'Total Near Collisions',
            'Total Hard Brakes',
            'Mean Near Collisions per Scenario',
            'Mean Hard Brakes per Scenario',
            '',
            '=== JAM PERSISTENCE ===',
            'Perpetual Jams Count',
            'Perpetual Jam Rate (%)',
        ],
        'Value': [
            '',
            f"{df['t_dissolve_numeric'].mean():.2f}",
            f"{df['t_dissolve_numeric'].median():.2f}",
            f"{df['t_dissolve_numeric'].std():.2f}",
            f"{df['t_dissolve_numeric'].min():.2f}",
            f"{df['t_dissolve_numeric'].max():.2f}",
            '',
            '',
            f"{df['final_throughput'].mean():.1f}",
            f"{df['final_throughput'].median():.1f}",
            f"{df['final_throughput'].std():.1f}",
            f"{df['final_throughput'].min():.0f}",
            f"{df['final_throughput'].max():.0f}",
            '',
            '',
            f"{df['wave_speed_numeric'].mean():.3f}",
            f"{df['wave_speed_numeric'].median():.3f}",
            f"{df['wave_speed_numeric'].std():.3f}",
            '',
            '',
            f"{df['max_stopped'].mean():.2f}",
            f"{df['max_stopped'].max():.0f}",
            '',
            '',
            f"{df['near_collisions'].sum():.0f}",
            f"{df['hard_brakes'].sum():.0f}",
            f"{df['near_collisions'].mean():.1f}",
            f"{df['hard_brakes'].mean():.1f}",
            '',
            '',
            f"{df['jam_perpetuo'].sum():.0f}",
            f"{df['jam_perpetuo'].mean() * 100:.1f}",
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
    df['wave_speed_numeric'] = pd.to_numeric(df['wave_speed'], errors='coerce')
    
    # Remove perpetual jams for "best" categories
    df_resolved = df[df['jam_perpetuo'] == 0]
    
    results = {
        'Category': [
            '=== BEST SCENARIOS ===',
            'Fastest Jam Dissolution',
            'Highest Throughput',
            'Fewest Stopped Vehicles',
            'Slowest Wave Propagation',
            'Safest (Fewest Events)',
            '',
            '=== WORST SCENARIOS ===',
            'Slowest Jam Dissolution',
            'Lowest Throughput',
            'Most Stopped Vehicles',
            'Fastest Wave Propagation',
            'Most Dangerous (Most Events)',
        ],
        'Scenario': [
            '',
            df_resolved.loc[df_resolved['t_dissolve_numeric'].idxmin(), 'nombre'],
            df.loc[df['final_throughput'].idxmax(), 'nombre'],
            df.loc[df['max_stopped'].idxmin(), 'nombre'],
            df_resolved.loc[df_resolved['wave_speed_numeric'].idxmin(), 'nombre'],
            df.loc[(df['near_collisions'] + df['hard_brakes']).idxmin(), 'nombre'],
            '',
            '',
            df_resolved.loc[df_resolved['t_dissolve_numeric'].idxmax(), 'nombre'],
            df.loc[df['final_throughput'].idxmin(), 'nombre'],
            df.loc[df['max_stopped'].idxmax(), 'nombre'],
            df_resolved.loc[df_resolved['wave_speed_numeric'].idxmax(), 'nombre'],
            df.loc[(df['near_collisions'] + df['hard_brakes']).idxmax(), 'nombre'],
        ],
        'Value': [
            '',
            f"{df_resolved['t_dissolve_numeric'].min():.1f}s",
            f"{df['final_throughput'].max():.0f} vehicles",
            f"{df['max_stopped'].min():.0f} vehicles",
            f"{df_resolved['wave_speed_numeric'].min():.3f} m/s",
            f"{(df['near_collisions'] + df['hard_brakes']).min():.0f} events",
            '',
            '',
            f"{df_resolved['t_dissolve_numeric'].max():.1f}s",
            f"{df['final_throughput'].min():.0f} vehicles",
            f"{df['max_stopped'].max():.0f} vehicles",
            f"{df_resolved['wave_speed_numeric'].max():.3f} m/s",
            f"{(df['near_collisions'] + df['hard_brakes']).max():.0f} events",
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
    df['wave_speed_numeric'] = pd.to_numeric(df['wave_speed'], errors='coerce')
    
    corr_cols = ['truck_pct', 'aggressive_pct', 't_dissolve_numeric', 
                 'wave_speed_numeric', 'final_throughput', 'max_stopped', 
                 'near_collisions', 'hard_brakes']
    
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation Coefficient'})
    
    # Rename labels for better readability
    labels = ['Truck %', 'Aggressive %', 'Dissolution Time', 'Wave Speed', 
              'Throughput', 'Max Stopped', 'Near Collisions', 'Hard Brakes']
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    
    plt.title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/16_correlation_matrix.png", dpi=300)
    plt.close()
    print("  ✅ Saved: 16_correlation_matrix.png")
    
    corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")


def perform_statistical_tests(df):
    """Perform statistical significance tests."""
    print("Generating: Statistical significance tests...")
    
    df['t_dissolve_numeric'] = pd.to_numeric(df['t_dissolve'], errors='coerce')
    
    results = []
    
    # Test 1: Does truck percentage significantly affect dissolution time?
    low_trucks = df[df['truck_pct'] <= 0.10]['t_dissolve_numeric'].dropna()
    high_trucks = df[df['truck_pct'] >= 0.50]['t_dissolve_numeric'].dropna()
    t_stat, p_val = stats.ttest_ind(low_trucks, high_trucks)
    results.append({
        'Test': 'Truck % Effect on Dissolution Time',
        'Comparison': '≤10% trucks vs ≥50% trucks',
        'T-statistic': f"{t_stat:.4f}",
        'P-value': f"{p_val:.6f}",
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
    
    # Test 2: Does aggression affect throughput?
    low_aggr = df[df['aggressive_pct'] <= 0.25]['final_throughput']
    high_aggr = df[df['aggressive_pct'] >= 0.75]['final_throughput']
    t_stat, p_val = stats.ttest_ind(low_aggr, high_aggr)
    results.append({
        'Test': 'Aggression Effect on Throughput',
        'Comparison': '≤25% aggressive vs ≥75% aggressive',
        'T-statistic': f"{t_stat:.4f}",
        'P-value': f"{p_val:.6f}",
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
    
    # Test 3: Safety events correlation with fleet composition
    corr, p_val = stats.pearsonr(df['truck_pct'], 
                                  df['near_collisions'] + df['hard_brakes'])
    results.append({
        'Test': 'Truck % vs Safety Events',
        'Comparison': 'Pearson correlation',
        'T-statistic': f"{corr:.4f}",
        'P-value': f"{p_val:.6f}",
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/statistical_tests.csv", index=False)
    print("  ✅ Saved: statistical_tests.csv")
    
    return results_df


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC ANALYSIS V2")
    print("=" * 70 + "\n")
    
    # Load master data
    master_df = load_master_data()
    if master_df is None:
        print("\n❌ Cannot proceed without master data file.")
        return
    
    print(f"\n📊 Generating visualizations...\n")
    
    # ── Master data heatmaps ──
    plot_all_heatmaps(master_df)
    
    # ── Trend analysis ──
    plot_truck_effect(master_df)
    plot_aggression_effect(master_df)
    
    # ── Select representative scenarios for time-series ──
    scenarios_to_compare = [
        'Exp_T00_A000',  # Baseline: all cars, all cautious
        'Exp_T00_A100',  # All cars, all aggressive
        'Exp_T25_A050',  # 25% trucks, mixed drivers
        'Exp_T50_A000',  # 50% trucks, all cautious
        'Exp_T50_A100',  # 50% trucks, all aggressive
        'Exp_T75_A050',  # 75% trucks, mixed
    ]
    
    # ── Time-series visualizations ──
    plot_velocity_evolution(scenarios_to_compare)
    plot_speed_variance_evolution(scenarios_to_compare)
    plot_gap_evolution(scenarios_to_compare)
    plot_vehicle_states_evolution(scenarios_to_compare[:4])  # Limit to 4 for readability
    plot_throughput_evolution(scenarios_to_compare)
    plot_safety_events_evolution(scenarios_to_compare)
    
    # ── Detailed analysis for extreme scenarios ──
    plot_detailed_scenario('Exp_T00_A000')  # Best case
    plot_detailed_scenario('Exp_T75_A100')  # Worst case
    plot_detailed_scenario('Exp_T25_A050')  # Realistic middle
    
    # ── Statistical analysis ──
    print(f"\n📈 Generating statistical summaries...\n")
    generate_summary_statistics(master_df)
    find_best_worst_scenarios(master_df)
    generate_correlation_analysis(master_df)
    perform_statistical_tests(master_df)
    
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
