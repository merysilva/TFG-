"""
analyze_focused.py — Focused Traffic Jam Analysis for Thesis
=============================================================
Analyzes controlled experiments with clear study-by-study comparisons.

Usage:
    python analyze_focused.py
    
Generates plots organized by study:
- Study 1: Density effects
- Study 2: Behavior optimization
- Study 3: Disturbance sensitivity
- Combined: Overall insights
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
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = "analysis3"
DATA_DIR = "resultados"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    if 'Density' in scenario_name:
        # Extract N value
        parts = scenario_name.split('_')
        for part in parts:
            if part.startswith('N') and len(part) > 1 and part[1:].isdigit():
                n = int(part[1:])
                return 'Density', n, scenario_name
    elif 'Behavior' in scenario_name:
        # Extract behavior type
        if 'Aggressive' in scenario_name:
            return 'Behavior', 1, scenario_name  # Ordering for plots
        elif 'Normal' in scenario_name:
            return 'Behavior', 2, scenario_name
        elif 'Conservative' in scenario_name:
            return 'Behavior', 3, scenario_name
    elif 'Disturbance' in scenario_name:
        # Extract D value (deceleration parameter)
        parts = scenario_name.split('_')
        for part in parts:
            # Look for D followed by digits (D08, D10, D12, D15, D18)
            if part.startswith('D') and len(part) > 1 and part[1:].isdigit():
                d = int(part[1:])
                return 'Disturbance', d, scenario_name
    
    # Fallback: categorize legacy scenario names (Base, HighSpeed, Safe, DEBUG, etc.)
    # Extract N value for potential density grouping
    n_value = None
    for part in parts:
        if part.startswith('N') and len(part) > 1 and part[1:].isdigit():
            n_value = int(part[1:])
            break
    
    # Check if it has behavior-specific parameters (S and T)
    has_s0 = any(part.startswith('S') and len(part) > 1 and 
                part[1:].replace('.','').isdigit() for part in parts)
    has_t = any(part.startswith('T') and len(part) > 1 and 
               part[1:].replace('.','').isdigit() for part in parts)
    
    if has_s0 or has_t:
        # Behavior study scenario
        if 'Safe' in scenario_name or 'Conservative' in scenario_name:
            return 'Behavior', 3, scenario_name  # Conservative
        elif 'Aggressive' in scenario_name or 'HighSpeed' in scenario_name:
            return 'Behavior', 1, scenario_name  # Aggressive
        else:
            return 'Behavior', 2, scenario_name  # Normal/Base
    
    elif n_value is not None:
        # Density-based grouping if N exists
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
        'velocity_drop': 0,  # Will calculate
        'total_energy': df['energia_disipada_acum'].max(),
        'total_time_lost': df['tiempo_perdido_acum'].max(),
        'min_efficiency': df['eficiencia_pct'].min(),
        'flow_before': df[df['segundo'] < 10]['flujo_veh_h'].mean(),
        'flow_during': df[(df['segundo'] >= 12) & (df['segundo'] <= 20)]['flujo_veh_h'].mean(),
        'flow_drop_pct': 0,  # Will calculate
    }
    
    # Calculate derived metrics
    metrics['velocity_drop'] = metrics['avg_velocity_before'] - metrics['avg_velocity_during']
    if metrics['flow_before'] > 0:
        metrics['flow_drop_pct'] = 100 * (metrics['flow_before'] - metrics['flow_during']) / metrics['flow_before']
    
    # Recovery time (simplified - when efficiency > 95% for 5+ seconds)
    for i in range(15, len(df) - 5):
        window = df.iloc[i:i+5]
        if (window['eficiencia_pct'].min() > 98 and window['coches_atasco'].max() < 2):  # Almost no vehicles in jam
        #if window['eficiencia_pct'].min() > 95:
            metrics['recovery_time'] = df.iloc[i]['segundo']
            break
    else:
        metrics['recovery_time'] = np.nan
    
    return metrics


# ═══════════════════════════════════════════════════════════════════
#  STUDY 1: DENSITY EFFECTS
# ═══════════════════════════════════════════════════════════════════

def plot_study1_density(density_scenarios):
    """Study 1: How density affects jam formation and recovery."""
    print("📊 Generating: Study 1 - Density Effects...")
    
    # Extract metrics
    data = []
    for n, (scenario_name, df) in sorted(density_scenarios.items()):
        metrics = extract_metrics(df, scenario_name)
        metrics['density'] = n
        data.append(metrics)
    
    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values('density')
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Jam severity (Max in Jam)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_df['density'], results_df['max_in_jam'], 
            'o-', linewidth=3, markersize=10, color='#E74C3C')
    ax1.set_xlabel('Number of Vehicles (N)', fontsize=12)
    ax1.set_ylabel('Max Vehicles in Jam', fontsize=12)
    ax1.set_title('Jam Severity vs Density', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recovery time
    ax2 = fig.add_subplot(gs[0, 1])
    recovered = results_df[results_df['recovery_time'].notna()]
    if len(recovered) > 0:
        ax2.plot(recovered['density'], recovered['recovery_time'], 
                'o-', linewidth=3, markersize=10, color='#3498DB')
        ax2.set_xlabel('Number of Vehicles (N)', fontsize=12)
        ax2.set_ylabel('Recovery Time (s)', fontsize=12)
        ax2.set_title('Recovery Duration vs Density', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gap pressure
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results_df['density'], results_df['max_gap_pressure'], 
            'o-', linewidth=3, markersize=10, color='#9B59B6')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
    ax3.set_xlabel('Number of Vehicles (N)', fontsize=12)
    ax3.set_ylabel('Max Gap Pressure', fontsize=12)
    ax3.set_title('Gap Compression vs Density', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity drop
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(results_df['density'], results_df['velocity_drop'], 
            'o-', linewidth=3, markersize=10, color='#E67E22')
    ax4.set_xlabel('Number of Vehicles (N)', fontsize=12)
    ax4.set_ylabel('Velocity Drop (m/s)', fontsize=12)
    ax4.set_title('Speed Reduction During Jam', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Flow capacity drop
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(results_df['density'], results_df['flow_drop_pct'], 
            'o-', linewidth=3, markersize=10, color='#16A085')
    ax5.set_xlabel('Number of Vehicles (N)', fontsize=12)
    ax5.set_ylabel('Flow Reduction (%)', fontsize=12)
    ax5.set_title('Capacity Drop During Jam', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Energy dissipation
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(results_df['density'], results_df['total_energy'], 
            'o-', linewidth=3, markersize=10, color='#F39C12')
    ax6.set_xlabel('Number of Vehicles (N)', fontsize=12)
    ax6.set_ylabel('Total Energy Lost (J)', fontsize=12)
    ax6.set_title('Energy Cost vs Density', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Time series comparison (select 3 scenarios)
    ax7 = fig.add_subplot(gs[2, :])
    densities_to_plot = [results_df['density'].min(), 
                        results_df['density'].median(), 
                        results_df['density'].max()]
    
    colors_ts = ['#2ECC71', '#F39C12', '#E74C3C']
    for i, n in enumerate(densities_to_plot):
        n_int = int(n)
        if n_int in density_scenarios:
            scenario_name, df = density_scenarios[n_int]
            ax7.plot(df['segundo'], df['coches_atasco'], 
                    linewidth=2.5, label=f'N={n_int}', color=colors_ts[i])
    
    ax7.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax7.set_xlabel('Time (seconds)', fontsize=12)
    ax7.set_ylabel('Vehicles in Jam State', fontsize=12)
    ax7.set_title('Jam Evolution: Low vs Medium vs High Density', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Study 1: Traffic Density Effects on Phantom Jam Formation', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f"{OUTPUT_DIR}/study1_density_effects.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: study1_density_effects.png")
    
    # Save data
    results_df.to_csv(f"{OUTPUT_DIR}/study1_density_data.csv", index=False)
    print("  ✅ Saved: study1_density_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  STUDY 2: BEHAVIOR OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_study2_behavior(behavior_scenarios):
    """Study 2: Which driving behavior minimizes stop-and-go?"""
    print("📊 Generating: Study 2 - Behavior Optimization...")
    
    behavior_order = ['Aggressive', 'Normal', 'Conservative']
    behavior_labels = {
        'Aggressive': 'Aggressive\n(s₀=2.0, T=0.8)',
        'Normal': 'Normal\n(s₀=2.5, T=1.0)',
        'Conservative': 'Conservative\n(s₀=4.0, T=1.5)'
    }
    
    # Extract metrics
    data = []
    for behavior_type in behavior_order:
        if behavior_type in behavior_scenarios:
            scenario_name, df = behavior_scenarios[behavior_type]
            metrics = extract_metrics(df, scenario_name)
            metrics['behavior'] = behavior_type
            data.append(metrics)
    
    results_df = pd.DataFrame(data)
    
    # Create comparison plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    x_pos = np.arange(len(behavior_order))
    
    # Plot 1: Jam severity
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x_pos, results_df['max_in_jam'], color=colors, alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([behavior_labels[b] for b in behavior_order], fontsize=10)
    ax1.set_ylabel('Max Vehicles in Jam', fontsize=12)
    ax1.set_title('Jam Severity by Behavior', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Recovery time
    ax2 = fig.add_subplot(gs[0, 1])
    recovery_data = [results_df[results_df['behavior'] == b]['recovery_time'].values[0] 
                    for b in behavior_order if not pd.isna(results_df[results_df['behavior'] == b]['recovery_time'].values[0])]
    if len(recovery_data) == len(behavior_order):
        bars2 = ax2.bar(x_pos, recovery_data, color=colors, alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([behavior_labels[b] for b in behavior_order], fontsize=10)
        ax2.set_ylabel('Recovery Time (s)', fontsize=12)
        ax2.set_title('Recovery Speed Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Gap pressure
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(x_pos, results_df['max_gap_pressure'], color=colors, alpha=0.8)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([behavior_labels[b] for b in behavior_order], fontsize=10)
    ax3.set_ylabel('Max Gap Pressure', fontsize=12)
    ax3.set_title('Gap Management Quality', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Energy efficiency
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(x_pos, results_df['total_energy'], color=colors, alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([behavior_labels[b] for b in behavior_order], fontsize=10)
    ax4.set_ylabel('Total Energy Lost (J)', fontsize=12)
    ax4.set_title('Energy Efficiency', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Velocity drop
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(x_pos, results_df['velocity_drop'], color=colors, alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([behavior_labels[b] for b in behavior_order], fontsize=10)
    ax5.set_ylabel('Velocity Drop (m/s)', fontsize=12)
    ax5.set_title('Speed Degradation', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Flow capacity
    ax6 = fig.add_subplot(gs[1, 2])
    bars6 = ax6.bar(x_pos, results_df['flow_drop_pct'], color=colors, alpha=0.8)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([behavior_labels[b] for b in behavior_order], fontsize=10)
    ax6.set_ylabel('Flow Reduction (%)', fontsize=12)
    ax6.set_title('Capacity Impact', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Time series comparison
    ax7 = fig.add_subplot(gs[2, :])
    for i, behavior_type in enumerate(behavior_order):
        if behavior_type in behavior_scenarios:
            scenario_name, df = behavior_scenarios[behavior_type]
            ax7.plot(df['segundo'], df['coches_atasco'], 
                    linewidth=3, label=behavior_labels[behavior_type].replace('\n', ' '), 
                    color=colors[i])
    
    ax7.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax7.set_xlabel('Time (seconds)', fontsize=12)
    ax7.set_ylabel('Vehicles in Jam State', fontsize=12)
    ax7.set_title('Jam Evolution: Behavior Comparison', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Study 2: Driver Behavior Optimization - Which Minimizes Stop-and-Go?', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f"{OUTPUT_DIR}/study2_behavior_optimization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: study2_behavior_optimization.png")
    
    # Save data
    results_df.to_csv(f"{OUTPUT_DIR}/study2_behavior_data.csv", index=False)
    print("  ✅ Saved: study2_behavior_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  STUDY 3: DISTURBANCE SENSITIVITY
# ═══════════════════════════════════════════════════════════════════

def plot_study3_disturbance(disturbance_scenarios):
    """Study 3: How initial disturbance strength affects jam formation."""
    print("📊 Generating: Study 3 - Disturbance Sensitivity...")
    
    # Extract metrics
    data = []
    for d_value, (scenario_name, df) in sorted(disturbance_scenarios.items()):
        metrics = extract_metrics(df, scenario_name)
        metrics['disturbance'] = d_value
        data.append(metrics)
    
    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values('disturbance')
    
    # Create plot
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Jam severity vs disturbance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_df['disturbance'], results_df['max_in_jam'], 
            'o-', linewidth=3, markersize=10, color='#E74C3C')
    ax1.set_xlabel('Disturbance Strength (m/s²)', fontsize=12)
    ax1.set_ylabel('Max Vehicles in Jam', fontsize=12)
    ax1.set_title('Jam Severity vs Disturbance', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recovery time vs disturbance
    ax2 = fig.add_subplot(gs[0, 1])
    recovered = results_df[results_df['recovery_time'].notna()]
    if len(recovered) > 0:
        ax2.plot(recovered['disturbance'], recovered['recovery_time'], 
                'o-', linewidth=3, markersize=10, color='#3498DB')
        ax2.set_xlabel('Disturbance Strength (m/s²)', fontsize=12)
        ax2.set_ylabel('Recovery Time (s)', fontsize=12)
        ax2.set_title('Recovery Duration vs Disturbance', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity drop vs disturbance
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results_df['disturbance'], results_df['velocity_drop'], 
            'o-', linewidth=3, markersize=10, color='#E67E22')
    ax3.set_xlabel('Disturbance Strength (m/s²)', fontsize=12)
    ax3.set_ylabel('Velocity Drop (m/s)', fontsize=12)
    ax3.set_title('Speed Impact vs Disturbance', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy vs disturbance
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(results_df['disturbance'], results_df['total_energy'], 
            'o-', linewidth=3, markersize=10, color='#9B59B6')
    ax4.set_xlabel('Disturbance Strength (m/s²)', fontsize=12)
    ax4.set_ylabel('Total Energy Lost (J)', fontsize=12)
    ax4.set_title('Energy Cost vs Disturbance', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Gap pressure vs disturbance
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(results_df['disturbance'], results_df['max_gap_pressure'], 
            'o-', linewidth=3, markersize=10, color='#1ABC9C')
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Disturbance Strength (m/s²)', fontsize=12)
    ax5.set_ylabel('Max Gap Pressure', fontsize=12)
    ax5.set_title('Gap Compression vs Disturbance', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Time series (all disturbances)
    ax6 = fig.add_subplot(gs[1, 2])
    colors_cmap = plt.cm.Reds(np.linspace(0.3, 0.9, len(disturbance_scenarios)))
    
    for i, (d_value, (scenario_name, df)) in enumerate(sorted(disturbance_scenarios.items())):
        ax6.plot(df['segundo'], df['coches_atasco'], 
                linewidth=2.5, label=f'D={d_value}', color=colors_cmap[i])
    
    ax6.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax6.set_xlabel('Time (seconds)', fontsize=12)
    ax6.set_ylabel('Vehicles in Jam State', fontsize=12)
    ax6.set_title('Jam Evolution by Disturbance Strength', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Study 3: Disturbance Sensitivity Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f"{OUTPUT_DIR}/study3_disturbance_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: study3_disturbance_sensitivity.png")
    
    # Save data
    results_df.to_csv(f"{OUTPUT_DIR}/study3_disturbance_data.csv", index=False)
    print("  ✅ Saved: study3_disturbance_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  COMBINED ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_combined_insights(all_scenarios):
    """Combined analysis across all studies."""
    print("📊 Generating: Combined Insights...")
    
    # Collect all metrics
    all_data = []
    for name, df in all_scenarios.items():
        study_type, param_value, _ = categorize_scenario(name)
        if study_type != 'Other':
            metrics = extract_metrics(df, name)
            metrics['study'] = study_type
            metrics['param_value'] = param_value
            all_data.append(metrics)
    
    combined_df = pd.DataFrame(all_data)
    
    # Create comprehensive comparison
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Plot 1: Correlation heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    
    corr_columns = ['max_in_jam', 'recovery_time', 'max_gap_pressure', 
                   'velocity_drop', 'total_energy', 'flow_drop_pct']
    corr_data = combined_df[corr_columns].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title('Correlation Matrix: Key Jam Metrics', fontsize=13, fontweight='bold')
    
    # Plot 2: Recovery time distribution by study
    ax2 = fig.add_subplot(gs[0, 2])
    recovered = combined_df[combined_df['recovery_time'].notna()]
    
    if len(recovered) > 0:
        study_types = recovered['study'].unique()
        data_by_study = [recovered[recovered['study'] == st]['recovery_time'].values 
                        for st in study_types]
        
        bp = ax2.boxplot(data_by_study, labels=study_types, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax2.set_ylabel('Recovery Time (s)', fontsize=12)
        ax2.set_title('Recovery Time by Study', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Jam severity vs gap pressure (all scenarios)
    ax3 = fig.add_subplot(gs[1, 0])
    
    colors_study = {'Density': '#E74C3C', 'Behavior': '#3498DB', 'Disturbance': '#2ECC71'}
    for study in combined_df['study'].unique():
        subset = combined_df[combined_df['study'] == study]
        ax3.scatter(subset['max_gap_pressure'], subset['max_in_jam'], 
                   s=150, alpha=0.7, label=study, color=colors_study.get(study, 'gray'))
    
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax3.set_xlabel('Max Gap Pressure', fontsize=12)
    ax3.set_ylabel('Max Vehicles in Jam', fontsize=12)
    ax3.set_title('Jam Severity vs Gap Compression', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy vs recovery time
    ax4 = fig.add_subplot(gs[1, 1])
    recovered = combined_df[combined_df['recovery_time'].notna()]
    
    if len(recovered) > 0:
        for study in recovered['study'].unique():
            subset = recovered[recovered['study'] == study]
            ax4.scatter(subset['recovery_time'], subset['total_energy'], 
                       s=150, alpha=0.7, label=study, color=colors_study.get(study, 'gray'))
        
        ax4.set_xlabel('Recovery Time (s)', fontsize=12)
        ax4.set_ylabel('Total Energy Lost (J)', fontsize=12)
        ax4.set_title('Energy Cost vs Recovery Duration', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Flow reduction comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    study_means = combined_df.groupby('study')['flow_drop_pct'].mean()
    bars = ax5.bar(range(len(study_means)), study_means.values, 
                   color=[colors_study.get(s, 'gray') for s in study_means.index])
    ax5.set_xticks(range(len(study_means)))
    ax5.set_xticklabels(study_means.index, fontsize=11)
    ax5.set_ylabel('Avg Flow Reduction (%)', fontsize=12)
    ax5.set_title('Capacity Drop by Study', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 6: Summary table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create summary statistics
    summary_text = []
    summary_text.append("═" * 100)
    summary_text.append("COMBINED ANALYSIS SUMMARY")
    summary_text.append("═" * 100)
    summary_text.append("")
    
    for study in ['Density', 'Behavior', 'Disturbance']:
        subset = combined_df[combined_df['study'] == study]
        if len(subset) > 0:
            summary_text.append(f"{'─' * 100}")
            summary_text.append(f"{study.upper()} STUDY ({len(subset)} scenarios)")
            summary_text.append(f"{'─' * 100}")
            summary_text.append(f"  Max in Jam:        {subset['max_in_jam'].min():.0f} - {subset['max_in_jam'].max():.0f} vehicles (avg: {subset['max_in_jam'].mean():.1f})")
            
            recovered = subset[subset['recovery_time'].notna()]
            if len(recovered) > 0:
                summary_text.append(f"  Recovery Time:     {recovered['recovery_time'].min():.0f} - {recovered['recovery_time'].max():.0f} seconds (avg: {recovered['recovery_time'].mean():.1f})")
            else:
                summary_text.append(f"  Recovery Time:     No recovery in simulation time")
            
            summary_text.append(f"  Gap Pressure:      {subset['max_gap_pressure'].min():.2f} - {subset['max_gap_pressure'].max():.2f} (avg: {subset['max_gap_pressure'].mean():.2f})")
            summary_text.append(f"  Velocity Drop:     {subset['velocity_drop'].min():.1f} - {subset['velocity_drop'].max():.1f} m/s (avg: {subset['velocity_drop'].mean():.1f})")
            summary_text.append(f"  Energy Lost:       {subset['total_energy'].min():.0f} - {subset['total_energy'].max():.0f} J (avg: {subset['total_energy'].mean():.0f})")
            summary_text.append(f"  Flow Reduction:    {subset['flow_drop_pct'].min():.1f} - {subset['flow_drop_pct'].max():.1f}% (avg: {subset['flow_drop_pct'].mean():.1f}%)")
            summary_text.append("")
    
    summary_text.append("═" * 100)
    summary_text.append("KEY FINDINGS:")
    summary_text.append("═" * 100)
    
    # Find best and worst scenarios
    best_recovery = combined_df[combined_df['recovery_time'].notna()].nsmallest(1, 'recovery_time')
    worst_jam = combined_df.nlargest(1, 'max_in_jam')
    least_energy = combined_df.nsmallest(1, 'total_energy')
    
    if len(best_recovery) > 0:
        summary_text.append(f"  ✓ Fastest Recovery:    {best_recovery.iloc[0]['scenario']} ({best_recovery.iloc[0]['recovery_time']:.0f}s)")
    if len(worst_jam) > 0:
        summary_text.append(f"  ✗ Most Severe Jam:     {worst_jam.iloc[0]['scenario']} ({worst_jam.iloc[0]['max_in_jam']:.0f} vehicles)")
    if len(least_energy) > 0:
        summary_text.append(f"  ✓ Most Efficient:      {least_energy.iloc[0]['scenario']} ({least_energy.iloc[0]['total_energy']:.0f}J)")
    
    # Add correlation insights
    summary_text.append("")
    summary_text.append(f"  Strongest Correlation: max_gap_pressure ↔ max_in_jam (r={corr_data.loc['max_gap_pressure', 'max_in_jam']:.2f})")
    
    # Display text
    text_str = '\n'.join(summary_text)
    ax6.text(0.05, 0.95, text_str, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Combined Analysis: Cross-Study Insights', 
                fontsize=16, fontweight='bold', y=0.998)
    
    plt.savefig(f"{OUTPUT_DIR}/combined_insights.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: combined_insights.png")
    
    # Save combined data
    combined_df.to_csv(f"{OUTPUT_DIR}/combined_data.csv", index=False)
    print("  ✅ Saved: combined_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "FOCUSED TRAFFIC JAM ANALYSIS" + " " * 33 + "║")
    print("║" + " " * 22 + "Thesis-Ready Plots" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Load all scenarios
    scenarios = load_all_scenarios()
    
    if not scenarios:
        print("\n❌ No scenario data found!")
        print(f"   Make sure CSV files exist in: {DATA_DIR}/")
        print("   Run sim.py or sim_focused.py first to generate data.\n")
        return
    
    print(f"\n📊 Loaded {len(scenarios)} scenarios")
    print(f"📁 Saving plots to: {OUTPUT_DIR}/\n")
    
    # Categorize scenarios by study
    density_scenarios = {}
    behavior_scenarios = {}
    disturbance_scenarios = {}
    
    for name, df in scenarios.items():
        study_type, param_value, scenario_name = categorize_scenario(name)
        
        if study_type == 'Density':
            density_scenarios[param_value] = (name, df)
        elif study_type == 'Behavior':
            behavior_name = 'Aggressive' if param_value == 1 else ('Normal' if param_value == 2 else 'Conservative')
            behavior_scenarios[behavior_name] = (name, df)
        elif study_type == 'Disturbance':
            disturbance_scenarios[param_value] = (name, df)
    
    print("=" * 80)
    print("GENERATING STUDY-SPECIFIC ANALYSES")
    print("=" * 80 + "\n")
    
    # Generate study-specific plots
    if len(density_scenarios) >= 3:
        plot_study1_density(density_scenarios)
    else:
        print("  ⚠️  Skipping Study 1: Need at least 3 density scenarios")
    
    if len(behavior_scenarios) >= 2:
        plot_study2_behavior(behavior_scenarios)
    else:
        print("  ⚠️  Skipping Study 2: Need at least 2 behavior scenarios")
    
    if len(disturbance_scenarios) >= 3:
        plot_study3_disturbance(disturbance_scenarios)
    else:
        print("  ⚠️  Skipping Study 3: Need at least 3 disturbance scenarios")
    
    # Generate combined analysis
    print("\n" + "=" * 80)
    print("GENERATING COMBINED ANALYSIS")
    print("=" * 80 + "\n")
    
    plot_combined_insights(scenarios)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Generated plots in: {OUTPUT_DIR}/")
    
    # Print file list
    print("\nGenerated files:")
    for f in sorted(Path(OUTPUT_DIR).glob("*")):
        print(f"  • {f.name}")
    print()


if __name__ == "__main__":
    main()