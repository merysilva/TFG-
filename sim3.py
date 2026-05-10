"""
simulador_heterogeneo.py — TFG Heterogeneous Traffic Flow
===========================================================
Simulates mixed traffic with different vehicle types and driver personalities.

NEW VARIABLES EXPLORED:
-----------------------
VEHICLE PROPERTIES:
    - vehicle_type: "car" (small, fast) or "truck" (large, slow)
    - length: physical size affecting spacing (4m for cars, 12m for trucks)
    - max_speed: vehicle speed limit (cars: 35 m/s, trucks: 25 m/s)
    - max_accel: acceleration capability (cars: 1.5, trucks: 0.8 m/s²)
    - max_decel: braking capability (cars: 3.0, trucks: 2.0 m/s²)

DRIVER PERSONALITY:
    - aggressiveness: 0.0-1.0 scale affecting:
        * desired_time_headway: aggressive=1.0s, cautious=2.5s
        * politeness: how much they accommodate others (0.0-0.5)
        * target_speed: percentage above speed limit they aim for

SIMULATION SCENARIOS:
    - fleet_composition: % of trucks (0%, 25%, 50%)
    - personality_mix: homogeneous vs heterogeneous drivers
    - disturbance_type: sudden braking event at t=60s

METRICS MEASURED:
-----------------
FLOW EFFICIENCY:
    - throughput: vehicles passing reference point per minute
    - average_speed: mean velocity across all vehicles
    - speed_variance: how much speeds differ (low = smooth flow)

SAFETY:
    - near_collisions: count when gap < 2m at speed > 5 m/s
    - hard_brakes: count of decelerations > 4 m/s²
    - min_gap_observed: closest approach between any two vehicles

Uso:
    python simulador_heterogeneo.py
"""

import pygame
import math
import csv
import random
import os

# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT SCENARIOS — vary fleet composition and personality mix
# ═══════════════════════════════════════════════════════════════════
ESCENARIOS = [
    # Pure car scenarios with different personality distributions
    {"nombre": "Exp_AllCars_Homogeneous", "truck_pct": 0.0, "personality_var": 0.1},
    {"nombre": "Exp_AllCars_Mixed", "truck_pct": 0.0, "personality_var": 0.5},
    {"nombre": "Exp_AllCars_Diverse", "truck_pct": 0.0, "personality_var": 0.9},
    
    # 25% trucks with varying personalities
    {"nombre": "Exp_25Trucks_Homogeneous", "truck_pct": 0.25, "personality_var": 0.1},
    {"nombre": "Exp_25Trucks_Mixed", "truck_pct": 0.25, "personality_var": 0.5},
    {"nombre": "Exp_25Trucks_Diverse", "truck_pct": 0.25, "personality_var": 0.9},
    
    # 50% trucks (heavy traffic mix)
    {"nombre": "Exp_50Trucks_Homogeneous", "truck_pct": 0.50, "personality_var": 0.1},
    {"nombre": "Exp_50Trucks_Mixed", "truck_pct": 0.50, "personality_var": 0.5},
    {"nombre": "Exp_50Trucks_Diverse", "truck_pct": 0.50, "personality_var": 0.9},
]

# ═══════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════
NUM_VEHICLES = 35                    # Total vehicles on track
TRACK_RADIUS = 400                   # meters
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS  # ~2513 meters
SIMULATION_DURATION = 300            # seconds to run (5 minutes)
DT = 0.1                            # physics timestep (seconds)

# Vehicle type specifications
VEHICLE_SPECS = {
    "car": {
        "length": 4.5,              # meters
        "max_speed": 35.0,          # m/s (~126 km/h)
        "max_accel": 1.5,           # m/s²
        "max_decel": 3.0,           # m/s²
        "color": (100, 150, 255),   # blue
    },
    "truck": {
        "length": 12.0,             # meters
        "max_speed": 25.0,          # m/s (~90 km/h)
        "max_accel": 0.8,           # m/s²
        "max_decel": 2.0,           # m/s²
        "color": (255, 150, 100),   # orange
    }
}

# IDM (Intelligent Driver Model) base parameters
IDM_S0 = 2.0                        # minimum gap (meters)
IDM_DELTA = 4                       # acceleration exponent

# Safety thresholds
COLLISION_THRESHOLD = 2.0           # gap < 2m = near collision
HARD_BRAKE_THRESHOLD = 4.0          # decel > 4 m/s² = hard brake


# ═══════════════════════════════════════════════════════════════════
#  VEHICLE CLASS — represents one car/truck with its own properties
# ═══════════════════════════════════════════════════════════════════
class Vehicle:
    def __init__(self, vehicle_id, vehicle_type, aggressiveness, position):
        """
        Initialize a vehicle with type-specific and personality-based properties.
        
        Args:
            vehicle_id: unique identifier (0, 1, 2, ...)
            vehicle_type: "car" or "truck"
            aggressiveness: 0.0 (cautious) to 1.0 (aggressive)
            position: starting position on track (meters)
        """
        self.id = vehicle_id
        self.type = vehicle_type
        self.aggressiveness = aggressiveness
        
        # Physical properties from VEHICLE_SPECS
        specs = VEHICLE_SPECS[vehicle_type]
        self.length = specs["length"]
        self.max_speed = specs["max_speed"]
        self.max_accel = specs["max_accel"]
        self.max_decel = specs["max_decel"]
        self.base_color = specs["color"]
        
        # Driver personality affects desired behavior
        # Aggressive drivers: shorter headway, higher target speed, less polite
        # Cautious drivers: longer headway, lower target speed, more polite
        self.desired_time_headway = 1.0 + (1.5 * (1.0 - aggressiveness))  # 1.0-2.5s
        self.politeness = 0.5 * (1.0 - aggressiveness)  # 0.0-0.5
        self.target_speed_factor = 0.9 + (0.2 * aggressiveness)  # 90-110% of max
        self.target_speed = self.max_speed * self.target_speed_factor
        
        # State variables
        self.position = position     # meters along track
        self.velocity = 0.0          # m/s (starts from rest)
        self.acceleration = 0.0      # m/s²
        
        # Statistics tracking
        self.near_collisions = 0
        self.hard_brakes = 0
        self.total_distance = 0.0
        self.min_gap_seen = float('inf')
    
    def get_display_color(self):
        """
        Color intensity based on speed — darker when slower.
        This helps visualize jams in the pygame window.
        """
        speed_ratio = self.velocity / self.target_speed if self.target_speed > 0 else 0
        speed_ratio = max(0.0, min(1.0, speed_ratio))  # clamp to [0,1]
        
        # Darken the base color when slow
        r = int(self.base_color[0] * (0.3 + 0.7 * speed_ratio))
        g = int(self.base_color[1] * (0.3 + 0.7 * speed_ratio))
        b = int(self.base_color[2] * (0.3 + 0.7 * speed_ratio))
        return (r, g, b)
    
    def calculate_idm_acceleration(self, gap, relative_velocity, lead_velocity):
        """
        Intelligent Driver Model (IDM) acceleration calculation.
        
        Args:
            gap: distance to vehicle ahead (meters)
            relative_velocity: self.velocity - lead.velocity (m/s)
            lead_velocity: velocity of vehicle ahead (m/s)
        
        Returns:
            desired acceleration (m/s²), can be negative (braking)
        """
        # Free-road acceleration term: accelerate toward target speed
        v_ratio = self.velocity / self.target_speed if self.target_speed > 0 else 0
        free_term = self.max_accel * (1.0 - v_ratio ** IDM_DELTA)
        
        # Desired dynamical spacing (depends on current speed and approach rate)
        interaction_term = self.velocity * self.desired_time_headway
        braking_term = (self.velocity * relative_velocity) / (2 * math.sqrt(self.max_accel * self.max_decel))
        s_star = IDM_S0 + max(0, interaction_term + braking_term)
        
        # Interaction term: brake to maintain safe gap
        gap = max(0.1, gap)  # prevent division by zero
        spacing_ratio = s_star / gap
        interaction = self.max_accel * (spacing_ratio ** 2)
        
        # Combined acceleration
        accel = free_term - interaction
        
        # Apply physical limits (can't accelerate/brake beyond vehicle capability)
        accel = max(-self.max_decel, min(self.max_accel, accel))
        
        return accel


# ═══════════════════════════════════════════════════════════════════
#  SIMULATOR CLASS — manages all vehicles and runs the experiment
# ═══════════════════════════════════════════════════════════════════
class SimuladorHeterogeneo:
    def __init__(self, config):
        """
        Initialize simulation with given scenario configuration.
        
        Args:
            config: dict with 'nombre', 'truck_pct', 'personality_var'
        """
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 900))
        pygame.display.set_caption(f"Heterogeneous Traffic — {config['nombre']}")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 18)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self.config = config
        self.sim_time = 0.0
        self.frame_count = 0
        
        # ── Create vehicle fleet ──────────────────────────────────
        self.vehicles = []
        truck_count = int(NUM_VEHICLES * config["truck_pct"])
        
        # Shuffle to randomize truck positions
        vehicle_types = (["truck"] * truck_count + 
                        ["car"] * (NUM_VEHICLES - truck_count))
        random.shuffle(vehicle_types)
        
        # Generate personality distribution
        # Lower variance = more similar drivers, higher = more diverse
        base_aggression = 0.5  # mean aggressiveness
        variance = config["personality_var"]
        
        spacing = TRACK_LENGTH / NUM_VEHICLES
        for i, vtype in enumerate(vehicle_types):
            # Generate aggressiveness with controlled variance
            aggression = base_aggression + random.gauss(0, variance * 0.3)
            aggression = max(0.0, min(1.0, aggression))  # clamp to [0,1]
            
            vehicle = Vehicle(
                vehicle_id=i,
                vehicle_type=vtype,
                aggressiveness=aggression,
                position=i * spacing
            )
            self.vehicles.append(vehicle)
        
        # ── Metrics tracking ──────────────────────────────────────
        self.second_logs = []           # time-series data (one row per second)
        self.reference_point = 0.0      # position to measure throughput
        self.vehicles_passed = 0        # count of vehicles passing reference
        self.last_check_positions = [v.position for v in self.vehicles]
        
        # ── Disturbance event (optional) ──────────────────────────
        # At t=60s, vehicle 0 will brake suddenly to create a jam
        self.disturbance_active = False
        self.disturbance_time = 60.0
        self.disturbance_duration = 8.0
    
    def step_physics(self):
        """
        Update vehicle positions and velocities for one timestep.
        This is the core simulation loop.
        """
        # ── Step 1: Calculate accelerations using IDM ────────────
        for i, vehicle in enumerate(self.vehicles):
            # Find the vehicle directly ahead (circular track)
            lead_idx = (i + 1) % NUM_VEHICLES
            lead = self.vehicles[lead_idx]
            
            # Calculate gap (accounting for vehicle lengths and wraparound)
            raw_gap = (lead.position - vehicle.position) % TRACK_LENGTH
            gap = raw_gap - lead.length
            
            # Relative velocity (positive = catching up to leader)
            relative_vel = vehicle.velocity - lead.velocity
            
            # Get IDM acceleration
            vehicle.acceleration = vehicle.calculate_idm_acceleration(
                gap, relative_vel, lead.velocity
            )
        
        # ── Step 2: Apply disturbance (if active) ────────────────
        if (self.disturbance_time <= self.sim_time < 
            self.disturbance_time + self.disturbance_duration):
            # Vehicle 0 brakes hard
            self.vehicles[0].acceleration = -6.0
            self.disturbance_active = True
        
        # ── Step 3: Integrate velocities and positions ───────────
        for vehicle in self.vehicles:
            # Update velocity
            vehicle.velocity += vehicle.acceleration * DT
            vehicle.velocity = max(0.0, min(vehicle.max_speed, vehicle.velocity))
            
            # Update position (with wraparound on circular track)
            vehicle.position += vehicle.velocity * DT
            vehicle.position %= TRACK_LENGTH
            
            # Track total distance traveled
            vehicle.total_distance += vehicle.velocity * DT
    
    def update_metrics(self):
        """
        Calculate safety and efficiency metrics.
        Called every timestep to track dangerous situations.
        """
        # ── Safety metrics ────────────────────────────────────────
        for i, vehicle in enumerate(self.vehicles):
            lead_idx = (i + 1) % NUM_VEHICLES
            lead = self.vehicles[lead_idx]
            
            gap = ((lead.position - vehicle.position) % TRACK_LENGTH) - lead.length
            
            # Track minimum gap seen
            if gap < vehicle.min_gap_seen:
                vehicle.min_gap_seen = gap
            
            # Near collision: gap < 2m while moving fast
            if gap < COLLISION_THRESHOLD and vehicle.velocity > 5.0:
                vehicle.near_collisions += 1
            
            # Hard braking event
            if vehicle.acceleration < -HARD_BRAKE_THRESHOLD:
                vehicle.hard_brakes += 1
    
    def log_second_data(self):
        """
        Record aggregate metrics for this second of simulation.
        Called every 1.0 simulated seconds (every 10 timesteps).
        """
        # ── Calculate fleet-wide statistics ──────────────────────
        velocities = [v.velocity for v in self.vehicles]
        avg_speed = sum(velocities) / len(velocities)
        speed_variance = sum((v - avg_speed)**2 for v in velocities) / len(velocities)
        min_speed = min(velocities)
        max_speed = max(velocities)
        
        # Throughput: count vehicles that crossed reference point this second
        crossed = 0
        for i, vehicle in enumerate(self.vehicles):
            old_pos = self.last_check_positions[i]
            new_pos = vehicle.position
            
            # Did vehicle cross reference point (accounting for wraparound)?
            if old_pos <= self.reference_point < new_pos:
                crossed += 1
            elif old_pos > new_pos and (old_pos <= self.reference_point or 
                                       new_pos >= self.reference_point):
                # Wraparound case
                crossed += 1
            
            self.last_check_positions[i] = new_pos
        
        self.vehicles_passed += crossed
        throughput_per_min = crossed * 60  # scale to vehicles/minute
        
        # ── Safety statistics ─────────────────────────────────────
        total_near_collisions = sum(v.near_collisions for v in self.vehicles)
        total_hard_brakes = sum(v.hard_brakes for v in self.vehicles)
        min_gaps = [v.min_gap_seen for v in self.vehicles if v.min_gap_seen < float('inf')]
        min_gap_overall = min(min_gaps) if min_gaps else 999.0
        
        # ── Vehicle type distribution ─────────────────────────────
        num_cars = sum(1 for v in self.vehicles if v.type == "car")
        num_trucks = NUM_VEHICLES - num_cars
        
        # ── Log the data ──────────────────────────────────────────
        self.second_logs.append({
            "second": int(self.sim_time),
            "avg_speed": round(avg_speed, 2),
            "speed_variance": round(speed_variance, 2),
            "min_speed": round(min_speed, 2),
            "max_speed": round(max_speed, 2),
            "throughput_per_min": throughput_per_min,
            "total_throughput": self.vehicles_passed,
            "near_collisions": total_near_collisions,
            "hard_brakes": total_hard_brakes,
            "min_gap": round(min_gap_overall, 2),
            "num_cars": num_cars,
            "num_trucks": num_trucks,
        })
    
    def draw(self):
        """
        Render the simulation visually using pygame.
        Shows circular track with colored vehicles and real-time stats.
        """
        self.screen.fill((15, 15, 20))
        
        # ── Draw track ────────────────────────────────────────────
        center_x, center_y = 500, 400
        pygame.draw.circle(self.screen, (40, 40, 45), (center_x, center_y), 
                          TRACK_RADIUS, 35)
        
        # ── Draw vehicles ─────────────────────────────────────────
        for vehicle in self.vehicles:
            angle = (vehicle.position / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = vehicle.get_display_color()
            
            # Size based on vehicle type
            if vehicle.type == "car":
                radius = 7
            else:  # truck
                radius = 11
            
            # Vehicle 0 (the one that brakes) is marked with white outline
            if vehicle.id == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 
                                 radius + 2)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        # ── Draw reference line (for throughput measurement) ──────
        ref_angle = (self.reference_point / TRACK_LENGTH) * 2 * math.pi
        ref_x1 = center_x + (TRACK_RADIUS - 50) * math.cos(ref_angle)
        ref_y1 = center_y + (TRACK_RADIUS - 50) * math.sin(ref_angle)
        ref_x2 = center_x + (TRACK_RADIUS + 50) * math.cos(ref_angle)
        ref_y2 = center_y + (TRACK_RADIUS + 50) * math.sin(ref_angle)
        pygame.draw.line(self.screen, (100, 255, 100), 
                        (int(ref_x1), int(ref_y1)), (int(ref_x2), int(ref_y2)), 2)
        
        # ── Display statistics panel ──────────────────────────────
        velocities = [v.velocity for v in self.vehicles]
        avg_speed = sum(velocities) / len(velocities)
        total_near = sum(v.near_collisions for v in self.vehicles)
        total_brakes = sum(v.hard_brakes for v in self.vehicles)
        
        info = [
            f"Scenario: {self.config['nombre']}",
            f"Time: {int(self.sim_time)}s / {SIMULATION_DURATION}s",
            f"",
            f"Fleet Composition:",
            f"  Cars:   {sum(1 for v in self.vehicles if v.type == 'car')}",
            f"  Trucks: {sum(1 for v in self.vehicles if v.type == 'truck')}",
            f"",
            f"Flow Metrics:",
            f"  Avg speed:     {avg_speed:.1f} m/s",
            f"  Throughput:    {self.vehicles_passed} vehicles",
            f"",
            f"Safety Metrics:",
            f"  Near collisions: {total_near}",
            f"  Hard brakes:     {total_brakes}",
            f"",
            f"[White circle = disturbance vehicle]",
        ]
        
        for idx, text in enumerate(info):
            if text == "":  # blank line
                continue
            surf = self.font_small.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 20 + idx * 22))
        
        # ── Color legend ──────────────────────────────────────────
        legend_y = 650
        pygame.draw.circle(self.screen, VEHICLE_SPECS["car"]["color"], 
                          (850, legend_y), 8)
        self.screen.blit(self.font_small.render("Car (fast, agile)", True, (200, 200, 200)),
                        (870, legend_y - 8))
        
        pygame.draw.circle(self.screen, VEHICLE_SPECS["truck"]["color"], 
                          (850, legend_y + 30), 12)
        self.screen.blit(self.font_small.render("Truck (slow, heavy)", True, (200, 200, 200)),
                        (870, legend_y + 22))
        
        pygame.display.flip()
    
    def save_csv(self):
        """
        Write time-series data to CSV file.
        One file per scenario with second-by-second statistics.
        """
        os.makedirs("resultados_heterogeneo", exist_ok=True)
        filename = f"resultados_heterogeneo/{self.config['nombre']}_series.csv"
        
        if not self.second_logs:
            print(f"  ⚠️  No data to save for {self.config['nombre']}")
            return
        
        keys = self.second_logs[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.second_logs)
        
        print(f"  ✅  Saved: {filename}")
    
    def run(self):
        """
        Main simulation loop.
        Runs for SIMULATION_DURATION seconds, updating physics at 10 Hz.
        """
        running = True
        
        while running and self.sim_time < SIMULATION_DURATION:
            # ── Handle pygame events ─────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
            
            # ── Physics step ──────────────────────────────────────
            self.step_physics()
            self.update_metrics()
            
            self.sim_time += DT
            self.frame_count += 1
            
            # ── Log data every second ─────────────────────────────
            if self.frame_count >= 10:  # 10 frames at dt=0.1 = 1 second
                self.log_second_data()
                self.frame_count = 0
            
            # ── Render at 30 FPS ──────────────────────────────────
            self.draw()
            self.clock.tick(30)
        
        # ── Simulation complete ───────────────────────────────────
        self.save_csv()
        return "NEXT"


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION — Run all scenarios
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC SIMULATION")
    print("=" * 70)
    print(f"  Total scenarios: {len(ESCENARIOS)}")
    print(f"  Vehicles per scenario: {NUM_VEHICLES}")
    print(f"  Track length: {TRACK_LENGTH:.0f} m")
    print(f"  Simulation duration: {SIMULATION_DURATION} s")
    print("=" * 70 + "\n")
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i}/{len(ESCENARIOS)}] Running: {config['nombre']}")
        print(f"  Truck %: {config['truck_pct']*100:.0f}%  "
              f"Personality variance: {config['personality_var']:.1f}")
        
        sim = SimuladorHeterogeneo(config)
        result = sim.run()
        
        if result == "QUIT":
            print("\n⛔  Simulation interrupted by user.")
            break
    
    pygame.quit()
    print("\n✅  All simulations complete!")
    print(f"     Data saved in: resultados_heterogeneo/\n")
