"""
analisis_heterogeneo.py — Analysis of Heterogeneous Traffic Data
==================================================================
Reads CSV files from resultados_heterogeneo/ and produces:
    1. Comparison plots: flow efficiency vs fleet composition
    2. Safety analysis: near-collisions and hard brakes
    3. Statistical summary: how truck % and personality variance affect outcomes

Uso:
    python analisis_heterogeneo.py
"""

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════
#  LOAD DATA FROM ALL SCENARIO CSV FILES
# ═══════════════════════════════════════════════════════════════════
def load_all_scenarios():
    """
    Load all scenario CSV files and parse their parameters.
    Returns: dict mapping scenario_name -> {data, truck_pct, personality_var}
    """
    data_dir = "resultados_heterogeneo"
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir}/ directory not found.")
        print("   Run simulador_heterogeneo.py first to generate data.")
        return {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*_series.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}/")
        return {}
    
    scenarios = {}
    for filepath in csv_files:
        # Parse scenario name from filename
        # Example: "Exp_AllCars_Homogeneous_series.csv"
        filename = os.path.basename(filepath)
        scenario_name = filename.replace("_series.csv", "")
        
        # Extract parameters from scenario name
        # Format: Exp_<truck%>_<personality>
        if "AllCars" in scenario_name:
            truck_pct = 0.0
        elif "25Trucks" in scenario_name:
            truck_pct = 0.25
        elif "50Trucks" in scenario_name:
            truck_pct = 0.50
        else:
            truck_pct = None
        
        if "Homogeneous" in scenario_name:
            personality_var = 0.1
        elif "Mixed" in scenario_name:
            personality_var = 0.5
        elif "Diverse" in scenario_name:
            personality_var = 0.9
        else:
            personality_var = None
        
        # Load CSV data
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        scenarios[scenario_name] = {
            "data": data,
            "truck_pct": truck_pct,
            "personality_var": personality_var,
            "filepath": filepath
        }
    
    print(f"✅ Loaded {len(scenarios)} scenarios from {data_dir}/\n")
    return scenarios


# ═══════════════════════════════════════════════════════════════════
#  EXTRACT SUMMARY METRICS FROM TIME-SERIES DATA
# ═══════════════════════════════════════════════════════════════════
def calculate_summary_metrics(scenarios):
    """
    For each scenario, compute aggregate metrics over the full simulation.
    Returns: dict with summary statistics per scenario
    """
    summaries = {}
    
    for name, scenario in scenarios.items():
        data = scenario["data"]
        
        # Convert string columns to floats
        avg_speeds = [float(row["avg_speed"]) for row in data]
        speed_vars = [float(row["speed_variance"]) for row in data]
        throughputs = [float(row["throughput_per_min"]) for row in data]
        
        # Final values (end of simulation)
        final_row = data[-1]
        total_throughput = int(final_row["total_throughput"])
        total_near_collisions = int(final_row["near_collisions"])
        total_hard_brakes = int(final_row["hard_brakes"])
        min_gap = float(final_row["min_gap"])
        
        # Aggregate statistics
        mean_speed = np.mean(avg_speeds)
        mean_variance = np.mean(speed_vars)
        mean_throughput = np.mean(throughputs)
        
        # After disturbance (t > 60s) — how well did it recover?
        post_disturbance = [row for row in data if int(row["second"]) > 70]
        if post_disturbance:
            recovery_speed = np.mean([float(row["avg_speed"]) for row in post_disturbance])
        else:
            recovery_speed = mean_speed
        
        summaries[name] = {
            "truck_pct": scenario["truck_pct"],
            "personality_var": scenario["personality_var"],
            "mean_speed": mean_speed,
            "mean_variance": mean_variance,
            "mean_throughput": mean_throughput,
            "total_throughput": total_throughput,
            "near_collisions": total_near_collisions,
            "hard_brakes": total_hard_brakes,
            "min_gap": min_gap,
            "recovery_speed": recovery_speed,
        }
    
    return summaries


# ═══════════════════════════════════════════════════════════════════
#  PLOT 1: FLOW EFFICIENCY (average speed and throughput)
# ═══════════════════════════════════════════════════════════════════
def plot_flow_efficiency(summaries):
    """
    Bar charts comparing average speed and throughput across scenarios.
    Grouped by fleet composition, colored by personality variance.
    """
    # Organize data by truck% and personality
    truck_levels = [0.0, 0.25, 0.50]
    personality_levels = [0.1, 0.5, 0.9]
    personality_labels = ["Homogeneous", "Mixed", "Diverse"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    # Extract metrics in grid format
    speeds_grid = np.zeros((len(truck_levels), len(personality_levels)))
    throughput_grid = np.zeros((len(truck_levels), len(personality_levels)))
    
    for name, metrics in summaries.items():
        ti = truck_levels.index(metrics["truck_pct"])
        pi = personality_levels.index(metrics["personality_var"])
        speeds_grid[ti, pi] = metrics["mean_speed"]
        throughput_grid[ti, pi] = metrics["total_throughput"]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1A: Average speed
    ax = axes[0]
    x = np.arange(len(truck_levels))
    width = 0.25
    for pi, (pers_label, color) in enumerate(zip(personality_labels, colors)):
        offsets = x + (pi - 1) * width
        values = speeds_grid[:, pi]
        ax.bar(offsets, values, width, label=pers_label, color=color, 
               edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Fleet Composition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Flow Efficiency: Average Speed\nHigher = Better', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0% trucks\n(all cars)', '25% trucks', '50% trucks'])
    ax.legend(title='Driver Personality', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 1B: Total throughput
    ax = axes[1]
    for pi, (pers_label, color) in enumerate(zip(personality_labels, colors)):
        offsets = x + (pi - 1) * width
        values = throughput_grid[:, pi]
        ax.bar(offsets, values, width, label=pers_label, color=color,
               edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Fleet Composition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Vehicles Passed', fontsize=12, fontweight='bold')
    ax.set_title('Flow Efficiency: Throughput\nHigher = Better', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0% trucks', '25% trucks', '50% trucks'])
    ax.legend(title='Driver Personality', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis1_flow_efficiency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("→ analysis1_flow_efficiency.png")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 2: SAFETY METRICS (near-collisions and hard brakes)
# ═══════════════════════════════════════════════════════════════════
def plot_safety_metrics(summaries):
    """
    Bar charts showing safety-related incidents across scenarios.
    Lower values = safer traffic.
    """
    truck_levels = [0.0, 0.25, 0.50]
    personality_levels = [0.1, 0.5, 0.9]
    personality_labels = ["Homogeneous", "Mixed", "Diverse"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    # Extract metrics
    collisions_grid = np.zeros((len(truck_levels), len(personality_levels)))
    brakes_grid = np.zeros((len(truck_levels), len(personality_levels)))
    
    for name, metrics in summaries.items():
        ti = truck_levels.index(metrics["truck_pct"])
        pi = personality_levels.index(metrics["personality_var"])
        collisions_grid[ti, pi] = metrics["near_collisions"]
        brakes_grid[ti, pi] = metrics["hard_brakes"]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 2A: Near collisions
    ax = axes[0]
    x = np.arange(len(truck_levels))
    width = 0.25
    for pi, (pers_label, color) in enumerate(zip(personality_labels, colors)):
        offsets = x + (pi - 1) * width
        values = collisions_grid[:, pi]
        ax.bar(offsets, values, width, label=pers_label, color=color,
               edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Fleet Composition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Near Collisions (count)', fontsize=12, fontweight='bold')
    ax.set_title('Safety: Near Collisions\n(gap < 2m at speed > 5 m/s)\nLower = Safer', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0% trucks', '25% trucks', '50% trucks'])
    ax.legend(title='Driver Personality', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2B: Hard brakes
    ax = axes[1]
    for pi, (pers_label, color) in enumerate(zip(personality_labels, colors)):
        offsets = x + (pi - 1) * width
        values = brakes_grid[:, pi]
        ax.bar(offsets, values, width, label=pers_label, color=color,
               edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Fleet Composition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hard Brakes (count)', fontsize=12, fontweight='bold')
    ax.set_title('Safety: Hard Braking Events\n(decel > 4 m/s²)\nLower = Smoother', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0% trucks', '25% trucks', '50% trucks'])
    ax.legend(title='Driver Personality', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis2_safety_metrics.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("→ analysis2_safety_metrics.png")


# ═══════════════════════════════════════════════════════════════════
#  PLOT 3: TIME-SERIES COMPARISON (one scenario from each group)
# ═══════════════════════════════════════════════════════════════════
def plot_time_series(scenarios):
    """
    Show evolution of speed over time for representative scenarios.
    Demonstrates how disturbance propagates differently.
    """
    # Select 3 representative scenarios
    selected = [
        "Exp_AllCars_Homogeneous",
        "Exp_25Trucks_Mixed",
        "Exp_50Trucks_Diverse"
    ]
    
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for scenario_name, color in zip(selected, colors):
        if scenario_name not in scenarios:
            continue
        
        data = scenarios[scenario_name]["data"]
        times = [int(row["second"]) for row in data]
        speeds = [float(row["avg_speed"]) for row in data]
        
        ax.plot(times, speeds, color=color, linewidth=2, 
                label=scenario_name.replace("Exp_", "").replace("_", " "))
    
    # Mark disturbance period
    ax.axvspan(60, 68, color='red', alpha=0.1, label='Disturbance period')
    ax.axvline(60, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(68, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Speed Evolution Over Time\nEffect of Fleet Heterogeneity on Disturbance Recovery', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis3_time_series.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("→ analysis3_time_series.png")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
def print_summary_table(summaries):
    """
    Print formatted table of key metrics for all scenarios.
    """
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE — All Scenarios")
    print("=" * 100)
    print(f"{'Scenario':<30} {'Truck%':<8} {'Pers.Var':<10} {'Avg Speed':<12} "
          f"{'Throughput':<12} {'Collisions':<12} {'Hard Brakes':<12}")
    print("-" * 100)
    
    # Sort by truck% then personality variance
    sorted_scenarios = sorted(summaries.items(), 
                             key=lambda x: (x[1]["truck_pct"], x[1]["personality_var"]))
    
    for name, metrics in sorted_scenarios:
        print(f"{name:<30} {metrics['truck_pct']*100:<7.0f}% "
              f"{metrics['personality_var']:<10.1f} {metrics['mean_speed']:<12.2f} "
              f"{metrics['total_throughput']:<12} {metrics['near_collisions']:<12} "
              f"{metrics['hard_brakes']:<12}")
    
    print("=" * 100 + "\n")
    
    # Save to CSV
    with open('summary_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Truck%', 'Personality_Var', 'Avg_Speed', 
                        'Throughput', 'Near_Collisions', 'Hard_Brakes', 
                        'Min_Gap', 'Recovery_Speed'])
        for name, metrics in sorted_scenarios:
            writer.writerow([
                name, 
                metrics['truck_pct'], 
                metrics['personality_var'],
                round(metrics['mean_speed'], 2),
                metrics['total_throughput'],
                metrics['near_collisions'],
                metrics['hard_brakes'],
                round(metrics['min_gap'], 2),
                round(metrics['recovery_speed'], 2)
            ])
    print("→ summary_table.csv")


# ═══════════════════════════════════════════════════════════════════
#  KEY FINDINGS
# ═══════════════════════════════════════════════════════════════════
def analyze_findings(summaries):
    """
    Extract and print key insights from the data.
    """
    print("\n" + "=" * 100)
    print("  KEY FINDINGS")
    print("=" * 100)
    
    # Finding 1: Effect of truck percentage
    pure_cars = [m for m in summaries.values() if m["truck_pct"] == 0.0]
    heavy_trucks = [m for m in summaries.values() if m["truck_pct"] == 0.50]
    
    avg_speed_cars = np.mean([m["mean_speed"] for m in pure_cars])
    avg_speed_trucks = np.mean([m["mean_speed"] for m in heavy_trucks])
    speed_reduction = ((avg_speed_cars - avg_speed_trucks) / avg_speed_cars) * 100
    
    print(f"\n1. FLEET COMPOSITION IMPACT:")
    print(f"   - Pure car fleets: {avg_speed_cars:.2f} m/s average speed")
    print(f"   - 50% truck fleets: {avg_speed_trucks:.2f} m/s average speed")
    print(f"   - Speed reduction: {speed_reduction:.1f}%")
    print(f"   → Trucks significantly slow down overall traffic flow")
    
    # Finding 2: Effect of personality variance
    homogeneous = [m for m in summaries.values() if m["personality_var"] == 0.1]
    diverse = [m for m in summaries.values() if m["personality_var"] == 0.9]
    
    collisions_homo = np.mean([m["near_collisions"] for m in homogeneous])
    collisions_div = np.mean([m["near_collisions"] for m in diverse])
    
    print(f"\n2. DRIVER DIVERSITY IMPACT:")
    print(f"   - Homogeneous drivers: {collisions_homo:.1f} near-collisions (avg)")
    print(f"   - Diverse drivers: {collisions_div:.1f} near-collisions (avg)")
    if collisions_div > collisions_homo:
        print(f"   → Heterogeneous personalities increase safety risks")
    else:
        print(f"   → Heterogeneous personalities have minimal safety impact")
    
    # Finding 3: Worst and best scenarios
    worst_safety = max(summaries.items(), key=lambda x: x[1]["near_collisions"])
    best_flow = max(summaries.items(), key=lambda x: x[1]["mean_speed"])
    
    print(f"\n3. EXTREME SCENARIOS:")
    print(f"   - Worst safety: {worst_safety[0]}")
    print(f"     ({worst_safety[1]['near_collisions']} near-collisions)")
    print(f"   - Best flow: {best_flow[0]}")
    print(f"     ({best_flow[1]['mean_speed']:.2f} m/s average speed)")
    
    print("\n" + "=" * 100 + "\n")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC ANALYSIS")
    print("=" * 70 + "\n")
    
    # Load data
    scenarios = load_all_scenarios()
    if not scenarios:
        return
    
    # Calculate summary metrics
    summaries = calculate_summary_metrics(scenarios)
    
    # Generate plots
    print("Generating analysis plots...\n")
    plot_flow_efficiency(summaries)
    plot_safety_metrics(summaries)
    plot_time_series(scenarios)
    
    # Print results
    print_summary_table(summaries)
    analyze_findings(summaries)
    
    print("=" * 70)
    print("  ✅ Analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
