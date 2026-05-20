"""
analyze_traffic.py — Comprehensive Traffic Jam Analysis
========================================================
Analyzes gap-based traffic simulation data to visualize phantom jams,
stop-and-go waves, and optimization of traffic flow behavior.

Focuses on demonstrating:
1. How traffic jams form "for no apparent reason"
2. Stop-and-go wave propagation
3. Which behaviors minimize traffic disruption

Usage:
    python analyze_traffic.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = "analysis_plots"
DATA_DIR = "resultados"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_recovery_time(df, num_cars):
    """
    Robust recovery detection using multiple criteria.
    
    Recovery is when ALL of these are stable for 5+ consecutive seconds:
    1. Efficiency > 95% (near full speed)
    2. Very few stopped (< 5% of fleet)
    3. Low speed variance (v_diff < 3.0 m/s)
    4. Relaxed spacing (gap pressure < 0.6)
    """
    if len(df) < 20:  # Need enough data
        return np.nan
    
    # Look for stable recovery window
    for i in range(15, len(df) - 5):  # Start after disturbance (t>15)
        window = df.iloc[i:i+5]
        
        # All criteria must be met
        efficiency_ok = window['eficiencia_pct'].min() > 95
        stopped_ok = window['coches_parados'].max() < max(1, num_cars * 0.05)
        variance_ok = window['v_diff'].max() < 3.0
        pressure_ok = window['presion_gaps'].max() < 0.6
        
        if efficiency_ok and stopped_ok and variance_ok and pressure_ok:
            return df.iloc[i]['segundo']
    
    return np.nan


def debug_scenario(df, scenario_name, params):
    """Print diagnostic information about a scenario."""
    print(f"\n{'='*70}")
    print(f"🔍 DEBUG: {scenario_name}")
    print(f"{'='*70}")
    
    # Configuration
    print(f"📋 Configuration:")
    print(f"   N = {params.get('num_cars', '?')}, "
          f"V_max = {params.get('max_speed', '?')} m/s, "
          f"Disturbance = {params.get('disturbance', '?')} m/s²")
    
    # Jam severity
    print(f"\n🚦 Jam Severity:")
    print(f"   Max stopped:     {df['max_parados'].max():.0f} vehicles")
    print(f"   Max in jam:      {df['max_en_atasco'].max():.0f} vehicles")
    print(f"   Max v_diff:      {df['v_diff'].max():.1f} m/s")
    print(f"   Max gap pressure: {df['presion_gaps'].max():.2f}×")
    
    # System impact
    print(f"\n📊 System Impact:")
    before = df[df['segundo'] < 10]['vel_media'].mean()
    during_jam = df[(df['segundo'] >= 12) & (df['segundo'] <= 20)]['vel_media'].mean()
    after_40s = df[df['segundo'] >= 40]['vel_media'].mean() if len(df) >= 40 else np.nan
    
    print(f"   Before disturbance:  {before:.1f} m/s")
    print(f"   During jam (t=12-20): {during_jam:.1f} m/s (drop: {before-during_jam:.1f} m/s)")
    if not np.isnan(after_40s):
        print(f"   After 40s:           {after_40s:.1f} m/s")
    
    # Efficiency
    min_eff = df['eficiencia_pct'].min()
    print(f"   Min efficiency:      {min_eff:.1f}%")
    
    # Energy and time
    print(f"\n⚡ Costs:")
    print(f"   Total energy lost:   {df['energia_disipada_acum'].max():.0f} J")
    print(f"   Total time lost:     {df['tiempo_perdido_acum'].max():.0f} s")
    
    # Recovery
    recovery_time = find_recovery_time(df, params.get('num_cars', 30))
    if not np.isnan(recovery_time):
        print(f"\n✅ Recovery at t = {recovery_time:.0f}s")
    else:
        # Check if still recovering
        if df['eficiencia_pct'].iloc[-1] > 90:
            print(f"\n⏳ Recovering (eff = {df['eficiencia_pct'].iloc[-1]:.1f}% at end)")
        else:
            print(f"\n❌ No recovery (eff = {df['eficiencia_pct'].iloc[-1]:.1f}% at end)")
    
    print(f"{'='*70}\n")
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


def extract_parameters(scenario_name):
    """Extract parameters from scenario name."""
    # Example: Base_N25_V25_D10 or HighSpeed_N30_V30_D12
    parts = scenario_name.split('_')
    params = {}
    
    for part in parts:
        if part.startswith('N'):
            params['num_cars'] = int(part[1:])
        elif part.startswith('V'):
            params['max_speed'] = int(part[1:])
        elif part.startswith('D'):
            params['disturbance'] = int(part[1:])
        elif part.startswith('S'):
            params['s0'] = float(part[1:])
        elif part.startswith('T'):
            params['t_reaction'] = float(part[1:])
    
    # Extract scenario type
    if 'Base' in scenario_name:
        params['type'] = 'Baseline'
    elif 'HighSpeed' in scenario_name:
        params['type'] = 'High Speed'
    elif 'Safe' in scenario_name:
        params['type'] = 'Conservative'
    elif 'Extreme' in scenario_name:
        params['type'] = 'Extreme'
    elif 'VISUAL' in scenario_name:
        params['type'] = 'Demo'
    else:
        params['type'] = 'Other'
    
    return params


# ═══════════════════════════════════════════════════════════════════
#  PLOT 1: SPACE-TIME DIAGRAM (The Classic!)
# ═══════════════════════════════════════════════════════════════════

def plot_spacetime_diagram(scenarios):
    """
    Classic space-time diagram showing vehicle trajectories.
    This is THE visualization for understanding traffic waves.
    """
    print("📊 Generating: Space-time diagram...")
    
    # Select a representative scenario
    scenario_names = [s for s in scenarios.keys() if 'Base_N30' in s and 'V25' in s]
    if not scenario_names:
        scenario_names = list(scenarios.keys())[:1]
    
    for scenario_name in scenario_names[:1]:  # Just plot one for clarity
        df = scenarios[scenario_name]
        params = extract_parameters(scenario_name)
        
        # We need position data per vehicle - reconstruct from gaps
        # This is approximate since we only have aggregated data
        # But we can show the velocity field as a proxy
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        height_ratios=[3, 1])
        
        # Top: Velocity heatmap (space-time)
        time = df['segundo'].values
        vel = df['vel_media'].values
        
        # Create color map based on state distribution
        stopped_ratio = df['coches_parados'] / params.get('num_cars', 30)
        atasco_ratio = df['coches_atasco'] / params.get('num_cars', 30)
        
        # Plot velocity evolution
        ax1.fill_between(time, 0, vel, alpha=0.3, color='blue', label='Average Velocity')
        
        # Overlay jam regions
        for i in range(len(df)):
            if df.iloc[i]['coches_parados'] > 0:
                ax1.axvspan(time[i]-0.5, time[i]+0.5, alpha=0.3, color='red')
        
        ax1.axvline(x=10, color='black', linestyle='--', linewidth=2, 
                   label='Disturbance', alpha=0.7)
        ax1.set_xlabel('Time (seconds)', fontsize=13)
        ax1.set_ylabel('Average Velocity (m/s)', fontsize=13)
        ax1.set_title(f'Traffic Wave Propagation: {scenario_name}\n'
                     f'Red regions = Stopped vehicles (phantom jam formation)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Bottom: State distribution over time
        ax2.fill_between(time, 0, df['coches_parados'], 
                        alpha=0.7, color='red', label='Stopped')
        ax2.fill_between(time, df['coches_parados'], 
                        df['coches_parados'] + df['coches_atasco'],
                        alpha=0.7, color='orange', label='Traffic Jam')
        ax2.fill_between(time, 
                        df['coches_parados'] + df['coches_atasco'],
                        df['coches_parados'] + df['coches_atasco'] + df['coches_ajustando'],
                        alpha=0.7, color='yellow', label='Adjusting')
        
        ax2.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Time (seconds)', fontsize=13)
        ax2.set_ylabel('Number of Vehicles', fontsize=13)
        ax2.set_title('Vehicle State Distribution Over Time', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/01_spacetime_diagram_{scenario_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    print("  ✅ Saved: 01_spacetime_diagram")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 2: STOP-AND-GO OSCILLATIONS
# ═══════════════════════════════════════════════════════════════════

def plot_stop_and_go(scenarios):
    """
    Demonstrate the stop-and-go wave phenomenon.
    Shows how a single disturbance creates repeated stops.
    """
    print("📊 Generating: Stop-and-go oscillations...")
    
    # Compare different densities
    baseline_scenarios = {k: v for k, v in scenarios.items() if 'Base' in k}
    
    if len(baseline_scenarios) < 3:
        baseline_scenarios = dict(list(scenarios.items())[:3])
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (scenario_name, df) in enumerate(list(baseline_scenarios.items())[:3]):
        params = extract_parameters(scenario_name)
        
        # Velocity
        axes[0].plot(df['segundo'], df['vel_media'], 
                    linewidth=2.5, label=scenario_name, color=colors[idx])
        
        # Stopped vehicles (the "stop" in stop-and-go)
        axes[1].plot(df['segundo'], df['coches_parados'], 
                    linewidth=2.5, label=scenario_name, color=colors[idx])
        
        # Gap pressure (NEW metric!)
        axes[2].plot(df['segundo'], df['presion_gaps'], 
                    linewidth=2.5, label=scenario_name, color=colors[idx])
        axes[2].axhline(y=1.0, color='red', linestyle=':', alpha=0.5, 
                       label='Critical threshold' if idx == 0 else '')
    
    # Styling
    for ax in axes:
        ax.axvline(x=10, color='black', linestyle='--', linewidth=2, 
                  alpha=0.5, label='Disturbance start')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    axes[0].set_ylabel('Average Velocity (m/s)', fontsize=12)
    axes[0].set_title('Stop-and-Go Phenomenon: Velocity Oscillations', 
                     fontsize=14, fontweight='bold')
    
    axes[1].set_ylabel('Stopped Vehicles', fontsize=12)
    axes[1].set_title('Vehicles Coming to Complete Stop', 
                     fontsize=13, fontweight='bold')
    
    axes[2].set_ylabel('Gap Pressure (ratio)', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=13)
    axes[2].set_title('Gap Compression (>1.0 = Traffic jam condition)', 
                     fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_stop_and_go_oscillations.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 02_stop_and_go_oscillations")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 3: HEATMAP - STATE EVOLUTION
# ═══════════════════════════════════════════════════════════════════

def plot_state_heatmap(scenarios):
    """
    Heatmap showing how vehicle states evolve over time.
    Clearly shows jam formation and dissolution.
    """
    print("📊 Generating: Vehicle state heatmap...")
    
    scenario_name = list(scenarios.keys())[0]
    df = scenarios[scenario_name]
    
    # Create state matrix
    state_matrix = np.column_stack([
        df['coches_libre'],
        df['coches_ajustando'],
        df['coches_atasco'],
        df['coches_parados']
    ])
    
    # Normalize to percentages
    state_matrix_pct = state_matrix / state_matrix.sum(axis=1, keepdims=True) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Stacked area chart
    time = df['segundo'].values
    ax1.fill_between(time, 0, state_matrix_pct[:, 0], 
                     alpha=0.8, color='#4CAF50', label='Free Flow')
    ax1.fill_between(time, state_matrix_pct[:, 0], 
                     state_matrix_pct[:, 0] + state_matrix_pct[:, 1],
                     alpha=0.8, color='#FFC107', label='Adjusting')
    ax1.fill_between(time, 
                     state_matrix_pct[:, 0] + state_matrix_pct[:, 1],
                     state_matrix_pct[:, 0] + state_matrix_pct[:, 1] + state_matrix_pct[:, 2],
                     alpha=0.8, color='#FF5722', label='Traffic Jam')
    ax1.fill_between(time,
                     state_matrix_pct[:, 0] + state_matrix_pct[:, 1] + state_matrix_pct[:, 2],
                     100,
                     alpha=0.8, color='#B71C1C', label='Stopped')
    
    ax1.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Percentage of Vehicles (%)', fontsize=12)
    ax1.set_title(f'Vehicle State Distribution: {scenario_name}\n'
                 f'Evolution from free flow to jam and back',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='right', fontsize=11)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gap pressure evolution
    ax2.plot(time, df['presion_gaps'], linewidth=3, color='#E91E63', 
            label='Gap Pressure')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label='Jam threshold')
    ax2.fill_between(time, 1.0, df['presion_gaps'], 
                    where=(df['presion_gaps'] >= 1.0),
                    alpha=0.3, color='red', label='Jam condition')
    ax2.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time (seconds)', fontsize=13)
    ax2.set_ylabel('Gap Pressure (ratio)', fontsize=12)
    ax2.set_title('Gap Compression: When spacing becomes insufficient', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_state_evolution_heatmap.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 03_state_evolution_heatmap")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 4: WAVE PROPAGATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_wave_propagation(scenarios):
    """
    Show how the jam wave propagates backward through traffic.
    Key insight: jams travel upstream!
    """
    print("📊 Generating: Wave propagation analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenario_name = list(scenarios.keys())[0]
    df = scenarios[scenario_name]
    params = extract_parameters(scenario_name)
    
    # Top left: Velocity min/max envelope
    axes[0, 0].fill_between(df['segundo'], df['vel_min'], df['vel_max'],
                            alpha=0.3, color='blue', label='Velocity range')
    axes[0, 0].plot(df['segundo'], df['vel_media'], linewidth=2.5, 
                   color='darkblue', label='Average velocity')
    axes[0, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 0].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[0, 0].set_title('Velocity Spread: Min to Max', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top right: V_diff (speed variance)
    axes[0, 1].plot(df['segundo'], df['v_diff'], linewidth=2.5, color='#D32F2F')
    axes[0, 1].axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=2.0, color='orange', linestyle=':', linewidth=2, 
                      alpha=0.7, label='Recovery threshold')
    axes[0, 1].set_ylabel('Speed Variance (m/s)', fontsize=12)
    axes[0, 1].set_title('Speed Heterogeneity (Higher = More chaotic)', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom left: Wave speed
    axes[1, 0].plot(df['segundo'], df['vel_onda_kmh'], linewidth=2.5, 
                   color='#7B1FA2')
    axes[1, 0].axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_ylabel('Wave Speed (km/h)', fontsize=12)
    axes[1, 0].set_title('Shockwave Propagation Speed (Negative = Backward)', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom right: Cumulative energy dissipation
    axes[1, 1].plot(df['segundo'], df['energia_disipada_acum'], 
                   linewidth=2.5, color='#F57C00')
    axes[1, 1].axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_ylabel('Energy Dissipated (J)', fontsize=12)
    axes[1, 1].set_title('Cumulative Energy Lost to Braking', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Wave Propagation Dynamics: {scenario_name}', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_wave_propagation_analysis.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 04_wave_propagation_analysis")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 5: DENSITY COMPARISON
# ═══════════════════════════════════════════════════════════════════

def plot_density_comparison(scenarios):
    """
    Compare how different traffic densities affect jam formation.
    """
    print("📊 Generating: Density comparison...")
    
    # Group by density
    density_groups = {}
    for name, df in scenarios.items():
        params = extract_parameters(name)
        if 'num_cars' in params:
            density = params['num_cars']
            if density not in density_groups:
                density_groups[density] = []
            density_groups[density].append((name, df, params))
    
    if len(density_groups) < 2:
        print("  ⚠️  Not enough density variations to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(density_groups)))
    
    for idx, (density, scenarios_list) in enumerate(sorted(density_groups.items())):
        # Take first scenario of each density
        name, df, params = scenarios_list[0]
        color = colors[idx]
        
        # Velocity evolution
        axes[0, 0].plot(df['segundo'], df['vel_media'], linewidth=2.5, 
                       label=f'N={density}', color=color)
        
        # Stopped vehicles
        axes[0, 1].plot(df['segundo'], df['coches_parados'], linewidth=2.5,
                       label=f'N={density}', color=color)
        
        # Flow
        axes[1, 0].plot(df['segundo'], df['flujo_veh_h'], linewidth=2.5,
                       label=f'N={density}', color=color)
        
        # Efficiency
        axes[1, 1].plot(df['segundo'], df['eficiencia_pct'], linewidth=2.5,
                       label=f'N={density}', color=color)
    
    # Styling
    for ax in axes.flat:
        ax.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    axes[0, 0].set_ylabel('Average Velocity (m/s)', fontsize=12)
    axes[0, 0].set_title('Velocity Evolution by Density', fontsize=13, fontweight='bold')
    
    axes[0, 1].set_ylabel('Stopped Vehicles', fontsize=12)
    axes[0, 1].set_title('Jam Severity by Density', fontsize=13, fontweight='bold')
    
    axes[1, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_ylabel('Flow (vehicles/hour)', fontsize=12)
    axes[1, 0].set_title('System Throughput', fontsize=13, fontweight='bold')
    
    axes[1, 1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_ylabel('Efficiency (%)', fontsize=12)
    axes[1, 1].set_title('System Efficiency', fontsize=13, fontweight='bold')
    
    plt.suptitle('Traffic Density Effects on Phantom Jam Formation', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_density_comparison.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 05_density_comparison")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 6: BEHAVIOR COMPARISON (OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════

def plot_behavior_comparison(scenarios):
    """
    Compare different driver behaviors: Which minimizes stops?
    This addresses the optimization question in your thesis.
    """
    print("📊 Generating: Behavior comparison (optimization)...")
    
    # Group by behavior type
    behavior_groups = {}
    for name, df in scenarios.items():
        params = extract_parameters(name)
        behavior_type = params.get('type', 'Other')
        if behavior_type not in behavior_groups:
            behavior_groups[behavior_type] = []
        behavior_groups[behavior_type].append((name, df, params))
    
    if len(behavior_groups) < 2:
        print("  ⚠️  Not enough behavior types to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    color_map = {
        'Baseline': '#2196F3',
        'High Speed': '#FF5722',
        'Conservative': '#4CAF50',
        'Extreme': '#9C27B0'
    }
    
    summary_data = []
    
    for behavior, scenarios_list in behavior_groups.items():
        if behavior == 'Demo':
            continue
        
        # Average across scenarios of same type
        all_max_stopped = []
        all_recovery_time = []
        all_energy = []
        
        for name, df, params in scenarios_list:
            all_max_stopped.append(df['max_parados'].max())
            
            # Use robust recovery detection
            num_cars = params.get('num_cars', 30)
            recovery_time = find_recovery_time(df, num_cars)
            if not np.isnan(recovery_time):
                all_recovery_time.append(recovery_time)
            
            all_energy.append(df['energia_disipada_acum'].max())
        
        summary_data.append({
            'behavior': behavior,
            'avg_max_stopped': np.mean(all_max_stopped),
            'avg_recovery_time': np.mean(all_recovery_time) if all_recovery_time else np.nan,
            'avg_energy': np.mean(all_energy),
            'scenarios': len(scenarios_list)
        })
        
        # Plot time series for one representative scenario
        name, df, params = scenarios_list[0]
        color = color_map.get(behavior, '#607D8B')
        
        axes[0, 0].plot(df['segundo'], df['coches_parados'], linewidth=2.5,
                       label=behavior, color=color)
        
        axes[0, 1].plot(df['segundo'], df['tiempo_perdido_acum'], linewidth=2.5,
                       label=behavior, color=color)
        
        axes[1, 0].plot(df['segundo'], df['presion_gaps'], linewidth=2.5,
                       label=behavior, color=color)
        
        axes[1, 1].plot(df['segundo'], df['eficiencia_pct'], linewidth=2.5,
                       label=behavior, color=color)
    
    # Styling
    for ax in axes.flat:
        ax.axvline(x=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    axes[0, 0].set_ylabel('Stopped Vehicles', fontsize=12)
    axes[0, 0].set_title('Stop Minimization: Which behavior performs best?', 
                        fontsize=13, fontweight='bold')
    
    axes[0, 1].set_ylabel('Cumulative Time Lost (s)', fontsize=12)
    axes[0, 1].set_title('Time Efficiency', fontsize=13, fontweight='bold')
    
    axes[1, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_ylabel('Gap Pressure (ratio)', fontsize=12)
    axes[1, 0].set_title('Gap Management', fontsize=13, fontweight='bold')
    axes[1, 0].axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
    
    axes[1, 1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_ylabel('Efficiency (%)', fontsize=12)
    axes[1, 1].set_title('System Performance', fontsize=13, fontweight='bold')
    
    plt.suptitle('Driver Behavior Optimization: Minimizing Stop-and-Go', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_behavior_optimization.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 06_behavior_optimization")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{OUTPUT_DIR}/behavior_summary.csv", index=False)
    print("  ✅ Saved: behavior_summary.csv")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 7: RECOVERY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_recovery_analysis(scenarios):
    """
    Analyze how quickly jams dissolve under different conditions.
    """
    print("📊 Generating: Recovery analysis...")
    
    recovery_data = []
    
    for name, df in scenarios.items():
        params = extract_parameters(name)
        num_cars = params.get('num_cars', 30)
        
        # Use robust recovery detection
        recovery_time = find_recovery_time(df, num_cars)
        
        # Peak jam severity
        max_stopped = df['max_parados'].max()
        max_atasco = df['max_en_atasco'].max()
        
        recovery_data.append({
            'scenario': name,
            'num_cars': num_cars,
            'max_speed': params.get('max_speed', 0),
            'type': params.get('type', 'Other'),
            'recovery_time': recovery_time,
            'max_stopped': max_stopped,
            'max_in_jam': max_atasco,
            'total_energy': df['energia_disipada_acum'].max()
        })
    
    recovery_df = pd.DataFrame(recovery_data)
    
    # Separate recovered and non-recovered
    recovered = recovery_df[recovery_df['recovery_time'].notna()]
    not_recovered = recovery_df[recovery_df['recovery_time'].isna()]
    
    if len(recovered) == 0:
        print("  ⚠️  No scenarios recovered within simulation time")
        print(f"      {len(not_recovered)} scenarios still jammed at end")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Recovery time vs density
    if len(recovered) > 0 and 'num_cars' in recovered.columns:
        for behavior_type in recovered['type'].unique():
            subset = recovered[recovered['type'] == behavior_type]
            axes[0, 0].scatter(subset['num_cars'], subset['recovery_time'],
                             s=150, alpha=0.7, label=behavior_type)
        
        axes[0, 0].set_xlabel('Number of Vehicles', fontsize=12)
        axes[0, 0].set_ylabel('Recovery Time (s)', fontsize=12)
        axes[0, 0].set_title('Recovery Time vs Traffic Density', 
                            fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Recovery time vs max stopped
    if len(recovered) > 0:
        axes[0, 1].scatter(recovered['max_stopped'], recovered['recovery_time'],
                          s=150, alpha=0.7, c=recovered['num_cars'], cmap='viridis')
        axes[0, 1].set_xlabel('Max Stopped Vehicles', fontsize=12)
        axes[0, 1].set_ylabel('Recovery Time (s)', fontsize=12)
        axes[0, 1].set_title('Jam Severity vs Recovery Duration', 
                            fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        if len(axes[0, 1].collections) > 0:
            cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
            cbar.set_label('Density (N)', fontsize=10)
    
    # Energy dissipated vs recovery time
    if len(recovered) > 0:
        axes[1, 0].scatter(recovered['total_energy'], recovered['recovery_time'],
                          s=150, alpha=0.7, c=recovered['max_stopped'], cmap='Reds')
        axes[1, 0].set_xlabel('Total Energy Dissipated (J)', fontsize=12)
        axes[1, 0].set_ylabel('Recovery Time (s)', fontsize=12)
        axes[1, 0].set_title('Energy Cost of Jam Recovery', 
                            fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        if len(axes[1, 0].collections) > 0:
            cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
            cbar.set_label('Max Stopped', fontsize=10)
    
    # Box plot by behavior type
    if len(recovered) > 0:
        behavior_types = recovered['type'].unique()
        data_by_type = [recovered[recovered['type'] == bt]['recovery_time'].values 
                        for bt in behavior_types]
        
        bp = axes[1, 1].boxplot(data_by_type, labels=behavior_types, patch_artist=True,
                                showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        axes[1, 1].set_ylabel('Recovery Time (s)', fontsize=12)
        axes[1, 1].set_title('Recovery Time Distribution by Behavior Type', 
                            fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Jam Recovery Analysis: What determines recovery speed?\n'
                f'{len(recovered)} scenarios recovered, {len(not_recovered)} still jammed', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_recovery_analysis.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 07_recovery_analysis")
    
    # Save recovery data
    recovery_df.to_csv(f"{OUTPUT_DIR}/recovery_data.csv", index=False)
    print("  ✅ Saved: recovery_data.csv")
    
    if len(not_recovered) > 0:
        print(f"  ℹ️  Note: {len(not_recovered)} scenarios did not fully recover")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 8: PHASE DIAGRAM
# ═══════════════════════════════════════════════════════════════════

def plot_phase_diagram(scenarios):
    """
    Phase space diagram: Density vs Flow, showing jam formation regions.
    """
    print("📊 Generating: Traffic flow phase diagram...")
    
    phase_data = []
    
    for name, df in scenarios.items():
        params = extract_parameters(name)
        
        # Take a snapshot before, during, and after jam
        for phase, time_range in [('Before', (0, 10)), 
                                   ('During', (15, 25)), 
                                   ('After', (40, 50))]:
            subset = df[(df['segundo'] >= time_range[0]) & 
                       (df['segundo'] <= time_range[1])]
            
            if len(subset) > 0:
                avg_density = subset['densidad_veh_km'].mean()
                avg_flow = subset['flujo_veh_h'].mean()
                avg_vel = subset['vel_media'].mean()
                
                phase_data.append({
                    'scenario': name,
                    'phase': phase,
                    'density': avg_density,
                    'flow': avg_flow,
                    'velocity': avg_vel,
                    'num_cars': params.get('num_cars', 0),
                    'type': params.get('type', 'Other')
                })
    
    phase_df = pd.DataFrame(phase_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Flow-Density diagram (fundamental diagram)
    phase_colors = {'Before': 'green', 'During': 'red', 'After': 'blue'}
    
    for phase in ['Before', 'During', 'After']:
        subset = phase_df[phase_df['phase'] == phase]
        axes[0].scatter(subset['density'], subset['flow'], 
                       s=150, alpha=0.7, label=phase, 
                       color=phase_colors[phase])
    
    axes[0].set_xlabel('Density (vehicles/km)', fontsize=12)
    axes[0].set_ylabel('Flow (vehicles/hour)', fontsize=12)
    axes[0].set_title('Fundamental Diagram: Flow vs Density\n'
                     'Red points = Jam state, Lower flow despite same density',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Velocity-Density diagram
    for phase in ['Before', 'During', 'After']:
        subset = phase_df[phase_df['phase'] == phase]
        axes[1].scatter(subset['density'], subset['velocity'], 
                       s=150, alpha=0.7, label=phase,
                       color=phase_colors[phase])
    
    axes[1].set_xlabel('Density (vehicles/km)', fontsize=12)
    axes[1].set_ylabel('Average Velocity (m/s)', fontsize=12)
    axes[1].set_title('Speed-Density Relationship\n'
                     'Shows capacity drop during jam',
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Traffic Flow Phase Diagram: Identifying Jam Conditions', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_phase_diagram.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 08_phase_diagram")
    
    # Save phase data
    phase_df.to_csv(f"{OUTPUT_DIR}/phase_data.csv", index=False)
    print("  ✅ Saved: phase_data.csv")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 9: SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════

def plot_summary_statistics(scenarios):
    """
    Generate summary statistics table and visualization.
    """
    print("📊 Generating: Summary statistics...")
    
    summary_data = []
    
    for name, df in scenarios.items():
        params = extract_parameters(name)
        num_cars = params.get('num_cars', 30)
        
        # Calculate key metrics
        max_stopped = df['max_parados'].max()
        max_in_jam = df['max_en_atasco'].max()
        
        # Use robust recovery detection
        recovery_time = find_recovery_time(df, num_cars)
        
        # Average during jam (t=15-25)
        jam_period = df[(df['segundo'] >= 15) & (df['segundo'] <= 25)]
        avg_vel_jam = jam_period['vel_media'].mean() if len(jam_period) > 0 else np.nan
        
        summary_data.append({
            'Scenario': name,
            'Type': params.get('type', 'Other'),
            'Density (N)': num_cars,
            'Max Speed': params.get('max_speed', 0),
            'Max Stopped': max_stopped,
            'Max in Jam': max_in_jam,
            'Recovery Time (s)': recovery_time,
            'Avg Velocity During Jam': avg_vel_jam,
            'Total Energy Lost (J)': df['energia_disipada_acum'].max(),
            'Total Time Lost (s)': df['tiempo_perdido_acum'].max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Max Stopped')
    
    # Save to CSV
    summary_df.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv", index=False)
    print("  ✅ Saved: summary_statistics.csv")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart: Max stopped by scenario
    top10 = summary_df.nsmallest(10, 'Max Stopped')
    axes[0, 0].barh(range(len(top10)), top10['Max Stopped'], color='#4CAF50')
    axes[0, 0].set_yticks(range(len(top10)))
    axes[0, 0].set_yticklabels(top10['Scenario'], fontsize=9)
    axes[0, 0].set_xlabel('Max Stopped Vehicles', fontsize=12)
    axes[0, 0].set_title('Top 10: Least Vehicles Stopped', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Bar chart: Recovery time (only recovered scenarios)
    recovered_df = summary_df[summary_df['Recovery Time (s)'].notna()]
    if len(recovered_df) > 0:
        top10_recovery = recovered_df.nsmallest(10, 'Recovery Time (s)')
        axes[0, 1].barh(range(len(top10_recovery)), top10_recovery['Recovery Time (s)'], 
                       color='#2196F3')
        axes[0, 1].set_yticks(range(len(top10_recovery)))
        axes[0, 1].set_yticklabels(top10_recovery['Scenario'], fontsize=9)
        axes[0, 1].set_xlabel('Recovery Time (s)', fontsize=12)
        axes[0, 1].set_title('Top 10: Fastest Recovery', 
                            fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    else:
        axes[0, 1].text(0.5, 0.5, 'No scenarios recovered\nwithin simulation time',
                       ha='center', va='center', fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Top 10: Fastest Recovery', fontsize=13, fontweight='bold')
    
    # Scatter: Recovery time vs max stopped (only recovered)
    if len(recovered_df) > 0:
        axes[1, 0].scatter(recovered_df['Max Stopped'], recovered_df['Recovery Time (s)'],
                          s=150, alpha=0.7, c=recovered_df['Density (N)'], cmap='plasma')
        axes[1, 0].set_xlabel('Max Stopped Vehicles', fontsize=12)
        axes[1, 0].set_ylabel('Recovery Time (s)', fontsize=12)
        axes[1, 0].set_title('Severity vs Recovery Duration', 
                            fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        if len(axes[1, 0].collections) > 0:
            cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
            cbar.set_label('Density', fontsize=10)
    else:
        axes[1, 0].text(0.5, 0.5, 'No recovery data available',
                       ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
    
    # Box plot by type
    types = summary_df['Type'].unique()
    data_by_type = [summary_df[summary_df['Type'] == t]['Max Stopped'].values 
                    for t in types]
    
    bp = axes[1, 1].boxplot(data_by_type, labels=types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    
    axes[1, 1].set_ylabel('Max Stopped Vehicles', fontsize=12)
    axes[1, 1].set_title('Jam Severity by Behavior Type', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Summary Statistics: Overall Performance Comparison', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_summary_statistics.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 09_summary_statistics")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 10: CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════

def plot_correlation_analysis(scenarios):
    """
    Analyze correlations between parameters and outcomes.
    """
    print("📊 Generating: Correlation analysis...")
    
    # Collect all metrics
    all_metrics = []
    
    for name, df in scenarios.items():
        params = extract_parameters(name)
        num_cars = params.get('num_cars', 30)
        
        # Use robust recovery detection
        recovery_time = find_recovery_time(df, num_cars)
        
        all_metrics.append({
            'num_cars': num_cars,
            'max_speed': params.get('max_speed', 0),
            's0': params.get('s0', 2.5),
            't_reaction': params.get('t_reaction', 1.0),
            'max_stopped': df['max_parados'].max(),
            'max_in_jam': df['max_en_atasco'].max(),
            'recovery_time': recovery_time,
            'total_energy': df['energia_disipada_acum'].max(),
            'time_lost': df['tiempo_perdido_acum'].max(),
            'max_gap_pressure': df['presion_gaps'].max()
        })
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # For correlation, only use scenarios that recovered
    # (NaN recovery times will be excluded automatically)
    corr = metrics_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Matrix: How parameters affect outcomes\n'
                'Red = Positive correlation, Blue = Negative correlation',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_correlation_matrix.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 10_correlation_matrix")
    
    # Save correlation data
    corr.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
    print("  ✅ Saved: correlation_matrix.csv")
    
    # Print key insights
    print("\n  📈 Key Correlations:")
    if 'num_cars' in corr.columns and 'max_gap_pressure' in corr.columns:
        print(f"     num_cars ↔ max_gap_pressure: {corr.loc['num_cars', 'max_gap_pressure']:.2f}")
    if 'num_cars' in corr.columns and 'max_in_jam' in corr.columns:
        print(f"     num_cars ↔ max_in_jam: {corr.loc['num_cars', 'max_in_jam']:.2f}")
    if 's0' in corr.columns and 'max_speed' in corr.columns:
        print(f"     s0 ↔ max_speed: {corr.loc['s0', 'max_speed']:.2f}")



# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "TRAFFIC JAM ANALYSIS SUITE" + " " * 32 + "║")
    print("║" + " " * 25 + "Gap-Based Simulator" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Load all scenarios
    scenarios = load_all_scenarios()
    
    if not scenarios:
        print("\n❌ No scenario data found!")
        print(f"   Make sure CSV files exist in: {DATA_DIR}/")
        print("   Run sim.py first to generate data.\n")
        return
    
    print(f"\n📊 Loaded {len(scenarios)} scenarios")
    print(f"📁 Saving plots to: {OUTPUT_DIR}/\n")
    
    # Debug first 3 scenarios to verify jam formation
    print("=" * 80)
    print("DEBUGGING SCENARIO DATA")
    print("=" * 80)
    
    for name, df in list(scenarios.items())[:3]:
        params = extract_parameters(name)
        debug_scenario(df, name, params)
    
    # Generate all plots
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    plot_spacetime_diagram(scenarios)
    plot_stop_and_go(scenarios)
    plot_state_heatmap(scenarios)
    plot_wave_propagation(scenarios)
    plot_density_comparison(scenarios)
    plot_behavior_comparison(scenarios)
    plot_recovery_analysis(scenarios)
    plot_phase_diagram(scenarios)
    plot_summary_statistics(scenarios)
    plot_correlation_analysis(scenarios)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Generated {len(list(Path(OUTPUT_DIR).glob('*.png')))} plots")
    print(f"📁 Results saved in: {OUTPUT_DIR}/\n")
    
    # Print file list
    print("Generated files:")
    for f in sorted(Path(OUTPUT_DIR).glob("*")):
        print(f"  • {f.name}")
    print()


if __name__ == "__main__":
    main()
