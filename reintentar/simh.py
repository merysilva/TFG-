"""
sim_heterogeneous.py — Heterogeneous Traffic Simulator (Cars + Trucks)
========================================================================
Extension of gap-based simulator to include trucks alongside cars.

KEY DIFFERENCE FROM HOMOGENEOUS:
- Vehicles have a "type" attribute: 'car' or 'truck'
- Trucks have different parameters: slower, larger gaps, slower reactions
- Scenarios vary ONLY by truck percentage (0%, 25%, 50%, 75%)

Usage:
    python sim_heterogeneous.py
"""

import pygame
import math
import csv
import time
import os
import random

# ═══════════════════════════════════════════════════════════════════════════
#  HETEROGENEOUS SCENARIOS - Varying Truck Percentage
# ═══════════════════════════════════════════════════════════════════════════

ESCENARIOS = [
    # ========================================================================
    # HETEROGENEOUS STUDY: Effect of Truck Percentage
    # Fixed: N=40, disturbance=-12, all else constant
    # Variable: truck_pct (0%, 10%, 25%, 50%, 75%)
    # ========================================================================
    
    {
        "nombre": "Het_Trucks_00pct",
        "num_vehicles": 40,
        "truck_pct": 0.00,  # 0% trucks (40 cars, 0 trucks) - BASELINE
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "Het_Trucks_10pct",
        "num_vehicles": 40,
        "truck_pct": 0.10,  # 10% trucks (36 cars, 4 trucks)
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "Het_Trucks_25pct",
        "num_vehicles": 40,
        "truck_pct": 0.25,  # 25% trucks (30 cars, 10 trucks)
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "Het_Trucks_50pct",
        "num_vehicles": 40,
        "truck_pct": 0.50,  # 50% trucks (20 cars, 20 trucks)
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "Het_Trucks_75pct",
        "num_vehicles": 40,
        "truck_pct": 0.75,  # 75% trucks (10 cars, 30 trucks)
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    
    # Optional: Visual demonstration
    {
        "nombre": "Het_VISUAL_50pct",
        "num_vehicles": 40,
        "truck_pct": 0.50,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": True  # Watch STATE-BASED colors: Red/Orange/Yellow/Green
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  VEHICLE PARAMETERS - Cars vs Trucks
# ═══════════════════════════════════════════════════════════════════════════

VEHICLE_PARAMS = {
    'car': {
        'max_speed': 25.0,      # m/s
        's0': 2.5,              # Minimum gap (m)
        't_reaction': 1.0,      # Reaction time (s)
        'acc_max': 2.5,         # Max acceleration (m/s²)
        'dec_max': 4.5,         # Max deceleration (m/s²)
        'color': (100, 150, 255),  # Blue
    },
    'truck': {
        'max_speed': 20.0,      # m/s (slower!)
        's0': 4.0,              # Minimum gap (m) (larger!)
        't_reaction': 1.5,      # Reaction time (s) (slower!)
        'acc_max': 1.5,         # Max acceleration (m/s²) (weaker!)
        'dec_max': 4.0,         # Max deceleration (m/s²)
        'color': (255, 100, 100),  # Red
    }
}


# ═══════════════════════════════════════════════════════════════════════════
#  STATE CLASSIFICATION - GAP-BASED THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

STOPPED_THRESHOLD = 1.0            # m/s - below this = stopped
GAP_MULTIPLIER_ATASCO = 1.5        # gap < 1.5 * critical = ATASCO
GAP_MULTIPLIER_LIBRE = 2.5         # gap > 2.5 * critical = LIBRE


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TRACK_LENGTH = 2513  # meters (circular track)
SIM_DURATION = 300   # seconds (5 minutes to see full dynamics)
DT = 0.1             # time step (seconds)

OUTPUT_DIR = "resultados_heterogeneo"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  VEHICLE CLASS - Now with Type!
# ═══════════════════════════════════════════════════════════════════════════

class Vehicle:
    def __init__(self, position, vehicle_type='car'):
        self.type = vehicle_type
        self.position = position
        self.velocity = VEHICLE_PARAMS[vehicle_type]['max_speed'] * 0.85  # Start at 85% max
        self.acceleration = 0.0
        
        # Get type-specific parameters
        self.max_speed = VEHICLE_PARAMS[vehicle_type]['max_speed']
        self.s0 = VEHICLE_PARAMS[vehicle_type]['s0']
        self.t_reaction = VEHICLE_PARAMS[vehicle_type]['t_reaction']
        self.acc_max = VEHICLE_PARAMS[vehicle_type]['acc_max']
        self.dec_max = VEHICLE_PARAMS[vehicle_type]['dec_max']
        self.color = VEHICLE_PARAMS[vehicle_type]['color']
        
        # State tracking
        self.state = "LIBRE"
        
    def get_gap_to_leader(self, leader_pos):
        """Calculate gap to vehicle ahead (accounting for circular track)."""
        if leader_pos > self.position:
            gap = leader_pos - self.position
        else:
            gap = (TRACK_LENGTH - self.position) + leader_pos
        return gap
    
    def classify_state(self, gap):
        """Classify vehicle state based on gap pressure."""
        if self.velocity < STOPPED_THRESHOLD:
            return "PARADO"
        
        # Critical gap = minimum safe following distance at current speed
        critical_gap = self.s0 + self.t_reaction * self.velocity
        
        if gap < critical_gap * GAP_MULTIPLIER_ATASCO:
            return "ATASCO"
        elif gap < critical_gap * GAP_MULTIPLIER_LIBRE:
            return "AJUSTANDO"
        else:
            return "LIBRE"
    
    def idm_acceleration(self, gap, delta_v):
        """
        Intelligent Driver Model (IDM) acceleration.
        
        gap: distance to leader (m)
        delta_v: velocity difference (our_v - leader_v) (m/s)
        """
        # Desired gap
        s_star = self.s0 + max(0, self.velocity * self.t_reaction + 
                              (self.velocity * delta_v) / (2 * math.sqrt(self.acc_max * self.dec_max)))
        
        # IDM formula
        acc = self.acc_max * (1 - (self.velocity / self.max_speed)**4 - (s_star / gap)**2)
        
        # Clamp to vehicle limits
        return max(-self.dec_max, min(self.acc_max, acc))
    
    def update(self, dt, gap, delta_v, disturbance_active=False, disturbance_decel=0):
        """Update vehicle position and velocity."""
        # Apply disturbance to first vehicle if active
        if disturbance_active:
            self.acceleration = disturbance_decel
        else:
            self.acceleration = self.idm_acceleration(gap, delta_v)
        
        # Update velocity and position
        self.velocity = max(0, self.velocity + self.acceleration * dt)
        self.position = (self.position + self.velocity * dt) % TRACK_LENGTH
        
        # Update state
        self.state = self.classify_state(gap)


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(config):
    """Run a single heterogeneous traffic scenario."""
    
    print(f"\n{'='*70}")
    print(f"Running: {config['nombre']}")
    print(f"  Vehicles: {config['num_vehicles']} ({config['truck_pct']*100:.0f}% trucks)")
    print(f"{'='*70}")
    
    # Calculate vehicle distribution
    num_vehicles = config['num_vehicles']
    num_trucks = int(num_vehicles * config['truck_pct'])
    num_cars = num_vehicles - num_trucks
    
    print(f"  → {num_cars} cars, {num_trucks} trucks")
    
    # Create vehicles with alternating pattern for even distribution
    vehicles = []
    spacing = TRACK_LENGTH / num_vehicles
    
    # Create type list with even distribution
    vehicle_types = ['truck'] * num_trucks + ['car'] * num_cars
    random.shuffle(vehicle_types)  # Mix them up
    
    for i, vtype in enumerate(vehicle_types):
        pos = i * spacing
        vehicles.append(Vehicle(pos, vehicle_type=vtype))
    
    # Data recording
    csv_data = []
    
    # Pygame setup (if visual enabled)
    if config.get('enable_visual', False):
        pygame.init()
        WIDTH, HEIGHT = 1200, 800
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Heterogeneous Traffic: {config['nombre']}")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
    else:
        screen = None
    
    # Simulation loop
    current_time = 0.0
    step = 0
    
    # Tracking variables
    max_stopped = 0
    max_in_jam = 0
    max_v_diff = 0
    
    while current_time < SIM_DURATION:
        # Check for quit
        if screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        # Update vehicles
        for i, vehicle in enumerate(vehicles):
            leader_idx = (i + 1) % len(vehicles)
            leader = vehicles[leader_idx]
            
            gap = vehicle.get_gap_to_leader(leader.position)
            delta_v = vehicle.velocity - leader.velocity
            
            # Check if disturbance is active for first vehicle
            disturbance_active = (i == 0 and 
                                config['disturbance_start'] <= current_time < 
                                config['disturbance_start'] + config['disturbance_duration'])
            
            vehicle.update(DT, gap, delta_v, disturbance_active, 
                          config.get('disturbance_decel', -12.0))
        
        # Record data every second
        if step % int(1.0 / DT) == 0:
            # Calculate metrics
            velocities = [v.velocity for v in vehicles]
            states = [v.state for v in vehicles]
            
            cars = [v for v in vehicles if v.type == 'car']
            trucks = [v for v in vehicles if v.type == 'truck']
            
            car_velocities = [v.velocity for v in cars] if cars else [0]
            truck_velocities = [v.velocity for v in trucks] if trucks else [0]
            
            # State counts
            stopped = sum(1 for s in states if s == "PARADO")
            in_jam = sum(1 for s in states if s == "ATASCO")
            adjusting = sum(1 for s in states if s == "AJUSTANDO")
            free_flow = sum(1 for s in states if s == "LIBRE")
            
            max_stopped = max(max_stopped, stopped)
            max_in_jam = max(max_in_jam, in_jam)
            
            # Velocity statistics
            avg_vel = sum(velocities) / len(velocities)
            min_vel = min(velocities)
            max_vel = max(velocities)
            v_diff = max_vel - min_vel
            max_v_diff = max(max_v_diff, v_diff)
            
            # Separate stats for cars vs trucks
            car_mean_vel = sum(car_velocities) / len(car_velocities)
            truck_mean_vel = sum(truck_velocities) / len(truck_velocities)
            
            # Gap analysis
            gaps = [vehicles[i].get_gap_to_leader(vehicles[(i+1)%len(vehicles)].position) 
                   for i in range(len(vehicles))]
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            
            # Gap pressure (using average critical gap across fleet)
            avg_critical_gap = sum(v.s0 + v.t_reaction * v.velocity for v in vehicles) / len(vehicles)
            gap_pressure = avg_critical_gap / avg_gap if avg_gap > 0 else 0
            
            # Flow metrics
            density_veh_km = (len(vehicles) / TRACK_LENGTH) * 1000
            flow_veh_h = density_veh_km * avg_vel * 3.6  # Convert to veh/hour
            
            # Efficiency (velocity relative to maximum possible)
            max_possible_vel = sum(v.max_speed for v in vehicles) / len(vehicles)
            efficiency_pct = (avg_vel / max_possible_vel) * 100
            
            # Record
            csv_data.append({
                'segundo': int(current_time),
                'estado_sistema': 'ATASCO' if in_jam > 0 else 'LIBRE',
                'vel_media': avg_vel,
                'vel_min': min_vel,
                'vel_max': max_vel,
                'v_diff': v_diff,
                'car_mean_velocity': car_mean_vel,
                'truck_mean_velocity': truck_mean_vel,
                'densidad_veh_km': density_veh_km,
                'flujo_veh_h': flow_veh_h,
                'eficiencia_pct': efficiency_pct,
                'coches_parados': stopped,
                'coches_atasco': in_jam,
                'coches_ajustando': adjusting,
                'coches_libre': free_flow,
                'gap_min': min_gap,
                'gap_medio': avg_gap,
                'gap_max': max_gap,
                'gap_critico_medio': avg_critical_gap,
                'presion_gaps': gap_pressure,
                'max_parados': max_stopped,
                'max_en_atasco': max_in_jam,
                'max_v_diff': max_v_diff,
                'num_cars': num_cars,
                'num_trucks': num_trucks,
                'truck_pct': config['truck_pct']
            })
        
        # Visualization
        if screen and step % 2 == 0:  # Draw every other frame
            screen.fill((20, 20, 20))
            
            # Draw track
            center_x, center_y = WIDTH // 2, HEIGHT // 2
            radius = 300
            pygame.draw.circle(screen, (60, 60, 60), (center_x, center_y), radius, 2)
            
            # Draw vehicles
            for vehicle in vehicles:
                angle = (vehicle.position / TRACK_LENGTH) * 2 * math.pi - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                # Size based on type - LARGER difference for visibility
                size = 14 if vehicle.type == 'truck' else 7
                
                # Use state-based colors like homogeneous sim
                if vehicle.state == "PARADO":
                    color = (200, 0, 0)  # Red - stopped
                elif vehicle.state == "ATASCO":
                    color = (255, 100, 0)  # Orange - jam
                elif vehicle.state == "AJUSTANDO":
                    color = (255, 200, 0)  # Yellow - adjusting
                else:  # LIBRE
                    if vehicle.type == 'truck':
                        color = (100, 100, 255)  # Light blue - truck in free flow
                    else:
                        color = (0, 200, 0)  # Green - car in free flow
                
                pygame.draw.circle(screen, color, (int(x), int(y)), size)
            
            # Info display
            info_texts = [
                f"Time: {current_time:.1f}s / {SIM_DURATION}s",
                f"Vehicles: {num_cars} cars (small), {num_trucks} trucks (large)",
                f"Colors: Red=Stopped, Orange=Jam, Yellow=Adjusting, Green/Blue=Free",
                f"Avg Speed: {avg_vel:.1f} m/s",
                f"In Jam: {in_jam} | Stopped: {stopped}",
                f"Car avg: {car_mean_vel:.1f} m/s | Truck avg: {truck_mean_vel:.1f} m/s",
                f"Gap Pressure: {gap_pressure:.2f}",
            ]
            
            for i, text in enumerate(info_texts):
                surface = font.render(text, True, (200, 200, 200))
                screen.blit(surface, (20, 20 + i * 30))
            
            pygame.display.flip()
            clock.tick(30)
        
        current_time += DT
        step += 1
    
    # Save CSV
    csv_filename = f"{OUTPUT_DIR}/{config['nombre']}_data.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"  ✅ Saved: {csv_filename}")
    print(f"  📊 Max in jam: {max_in_jam} | Max stopped: {max_stopped}")
    
    if screen:
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "HETEROGENEOUS TRAFFIC SIMULATION" + " " * 31 + "║")
    print("║" + " " * 25 + "Cars + Trucks" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"🚗 Vehicle types: Cars (blue, fast) vs Trucks (red, slow)")
    print(f"📊 Running {len(ESCENARIOS)} scenarios...\n")
    
    start_time = time.time()
    
    for config in ESCENARIOS:
        run_scenario(config)
    
    elapsed = time.time() - start_time
    
    print("\n" + "═" * 80)
    print(f"✅ ALL SIMULATIONS COMPLETE!")
    print(f"   Time elapsed: {elapsed:.1f} seconds")
    print(f"   Results saved in: {OUTPUT_DIR}/")
    print("═" * 80 + "\n")
    
    print("Next steps:")
    print("  1. Run: python analyze_heterogeneous.py")
    print("  2. Check: resultados_heterogeneo/ for CSV data")
    print("  3. View: analysis_heterogeneous/ for plots\n")


if __name__ == "__main__":
    main()