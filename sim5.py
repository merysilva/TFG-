"""
simulador_heterogeneo_v2.py — Improved Heterogeneous Traffic Flow Simulation
=============================================================================
Enhanced version with:
    - Realistic vehicle differentiation (trucks vs cars)
    - Binary driver personality (aggressive vs cautious)
    - Comprehensive data logging with wave propagation metrics
    - No permanent jams through calibrated parameters
    - Optional visualization (ENABLE_VISUAL flag)

Uso:
    python simulador_heterogeneo_v2.py
"""

import pygame
import math
import csv
import random
import os
import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
ENABLE_VISUAL = False  # Set to False for faster data-only runs
VISUAL_SPEED = 30      # FPS for visualization (higher = faster)

# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
TRUCK_PERCENTAGES = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]
AGGRESSIVE_PERCENTAGES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Generate all scenario combinations
ESCENARIOS = []
for truck_pct in TRUCK_PERCENTAGES:
    for aggr_pct in AGGRESSIVE_PERCENTAGES:
        ESCENARIOS.append({
            "nombre": f"Exp_T{int(truck_pct*100):02d}_A{int(aggr_pct*100):03d}",
            "truck_pct": truck_pct,
            "aggressive_pct": aggr_pct
        })

# ═══════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS (Calibrated to avoid permanent jams)
# ═══════════════════════════════════════════════════════════════════
NUM_CARS = 30                        # Reduced from 38 to prevent permanent jams
TRACK_RADIUS = 400                   # meters
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS

# Vehicle specifications - cars vs trucks
CAR_LENGTH = 5.0                     # meters
TRUCK_LENGTH = 12.0                  # meters
CAR_MAX_SPEED = 30.0                 # m/s (108 km/h)
TRUCK_MAX_SPEED = 25.0               # m/s (90 km/h)
CAR_ACC_MAX = 1.5                    # m/s² (cars accelerate faster)
TRUCK_ACC_MAX = 0.8                  # m/s² (trucks accelerate slower)
CAR_DEC_MAX = 3.0                    # m/s² (cars can brake harder)
TRUCK_DEC_MAX = 2.0                  # m/s² (trucks brake slower - longer stopping distance)

# IDM base parameters
S0 = 5.0                            # minimum gap (meters)
T_REACTION_BASE = 1.5               # base reaction time (seconds)

# Driver personality parameters
# Aggressive: want to go faster, accept smaller gaps
AGGRESSIVE_SPEED_FACTOR = 1.0       # 100% of max speed
AGGRESSIVE_GAP_FACTOR = 0.8         # accept 20% smaller gaps
CAUTIOUS_SPEED_FACTOR = 0.85        # 85% of max speed
CAUTIOUS_GAP_FACTOR = 1.2           # want 20% larger gaps

# Disturbance parameters (softened to avoid permanent jams)
DISTURBANCE_START = 10.0            # seconds
DISTURBANCE_DURATION = 3.0          # seconds (reduced from 5.0)
DISTURBANCE_DECEL = -8.0            # m/s² (softened from -12.0)

# Simulation parameters
MAX_TIME = 300                      # seconds
DT = 0.1                           # timestep


# ═══════════════════════════════════════════════════════════════════
#  SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════
class SimuladorHeterogeneo:
    def __init__(self, config, enable_visual=True):
        """Initialize simulation with fleet composition from config."""
        self.enable_visual = enable_visual
        
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"TFG — {config['nombre']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 15)
        
        self.config = config
        
        # ── Create vehicle fleet ──────────────────────────────────
        num_trucks = int(NUM_CARS * config["truck_pct"])
        num_cars = NUM_CARS - num_trucks
        num_aggressive = int(NUM_CARS * config["aggressive_pct"])
        
        # Create type arrays and shuffle
        vehicle_types = (["truck"] * num_trucks + ["car"] * num_cars)
        random.shuffle(vehicle_types)
        
        # Create personality arrays and shuffle
        personalities = (["aggressive"] * num_aggressive + 
                        ["cautious"] * (NUM_CARS - num_aggressive))
        random.shuffle(personalities)
        
        # Initialize vehicle properties based on type and personality
        self.vehicle_types = vehicle_types
        self.personalities = personalities
        self.lengths = []
        self.max_speeds = []
        self.acc_maxes = []
        self.dec_maxes = []
        self.v_desired = []
        self.gap_factors = []
        
        for vtype, pers in zip(vehicle_types, personalities):
            # Physical properties based on vehicle type
            if vtype == "car":
                self.lengths.append(CAR_LENGTH)
                self.max_speeds.append(CAR_MAX_SPEED)
                self.acc_maxes.append(CAR_ACC_MAX)
                self.dec_maxes.append(CAR_DEC_MAX)
            else:  # truck
                self.lengths.append(TRUCK_LENGTH)
                self.max_speeds.append(TRUCK_MAX_SPEED)
                self.acc_maxes.append(TRUCK_ACC_MAX)
                self.dec_maxes.append(TRUCK_DEC_MAX)
            
            # Behavioral properties based on personality
            if pers == "aggressive":
                self.v_desired.append(self.max_speeds[-1] * AGGRESSIVE_SPEED_FACTOR)
                self.gap_factors.append(AGGRESSIVE_GAP_FACTOR)
            else:  # cautious
                self.v_desired.append(self.max_speeds[-1] * CAUTIOUS_SPEED_FACTOR)
                self.gap_factors.append(CAUTIOUS_GAP_FACTOR)
        
        # ── State variables ───────────────────────────────────────
        spacing = TRACK_LENGTH / NUM_CARS
        self.positions = [i * spacing for i in range(NUM_CARS)]
        self.velocities = [0.0] * NUM_CARS
        self.accelerations = [0.0] * NUM_CARS
        
        # ── Simulation state ──────────────────────────────────────
        self.estado = "ESTABILIZANDO"
        self.segundo_actual = 0
        self.frame_count = 0
        self.timer_frenada = 0
        self.timer_fin = 0
        
        # ── Metrics tracking ──────────────────────────────────────
        self.logs = []
        self.reference_point = 0.0
        self.vehicles_passed_total = 0
        self.last_positions = self.positions.copy()
        self.near_collisions_count = 0
        self.hard_brakes_count = 0
        self.t_dissolve = None
        self.max_stopped = 0
        self.jam_perpetuo = False
        
        # Wave propagation tracking
        self.wave_start_time = None
        self.wave_positions = []  # Track the jam wave position
        self.wave_speed = None
        
        # Speed variance tracking (by type)
        self.speed_variance_history = []

    def calculate_idm_acceleration(self, i):
        """Calculate IDM acceleration for vehicle i with personalized parameters."""
        lead = (i + 1) % NUM_CARS
        
        # Calculate gap
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - self.lengths[lead])
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        v_desired = self.v_desired[i]
        acc_max = self.acc_maxes[i]
        dec_max = self.dec_maxes[i]
        gap_factor = self.gap_factors[i]
        
        # Free flow term
        if v_desired > 0:
            v_ratio = v / v_desired
        else:
            v_ratio = 0
        free_term = acc_max * (1.0 - v_ratio ** 4)
        
        # Interaction term with personalized gap acceptance
        s_desired = S0 * gap_factor + max(0, v * T_REACTION_BASE + 
                                          (v * dv) / (2 * math.sqrt(acc_max * dec_max)))
        interaction = acc_max * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        
        # Limit acceleration/deceleration to vehicle capabilities
        accel = max(-dec_max, min(acc_max, accel))
        
        return accel

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        if self.estado == "ESTABILIZANDO":
            # Gentle acceleration to desired speed
            for i in range(NUM_CARS):
                if self.velocities[i] < self.v_desired[i]:
                    self.accelerations[i] = self.acc_maxes[i] * 0.5
                else:
                    self.accelerations[i] = 0.0
        else:
            # IDM control
            for i in range(NUM_CARS):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance
        if self.estado == "FRENANDO":
            self.accelerations[0] = DISTURBANCE_DECEL
        
        # Update velocities and positions
        for i in range(NUM_CARS):
            self.velocities[i] += self.accelerations[i] * DT
            self.velocities[i] = max(0.0, min(self.max_speeds[i], self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def update_metrics(self):
        """Calculate real-time safety and performance metrics."""
        # Safety metrics
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            
            # Near collision: gap < 2m while moving
            if gap < 2.0 and self.velocities[i] > 2.0:
                self.near_collisions_count += 1
            
            # Hard braking event
            if self.accelerations[i] < -4.0:
                self.hard_brakes_count += 1
        
        # Track maximum stopped vehicles
        stopped = sum(1 for v in self.velocities if v < 1.0)
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        # Wave propagation tracking (find the jam front)
        if self.estado in ["FRENANDO", "RECUPERANDO"]:
            if self.wave_start_time is None and stopped > 5:
                self.wave_start_time = self.segundo_actual
            
            if stopped > 5:
                # Find the position of the jam (average position of stopped vehicles)
                stopped_positions = [self.positions[i] for i in range(NUM_CARS) 
                                    if self.velocities[i] < 1.0]
                if stopped_positions:
                    avg_jam_pos = sum(stopped_positions) / len(stopped_positions)
                    self.wave_positions.append((self.segundo_actual, avg_jam_pos))

    def calculate_wave_speed(self):
        """Calculate backward propagation speed of the jam wave."""
        if len(self.wave_positions) < 2:
            return None
        
        # Take first and last measurements
        t1, pos1 = self.wave_positions[0]
        t2, pos2 = self.wave_positions[-1]
        
        # Calculate distance traveled (accounting for circular track)
        distance = (pos1 - pos2) % TRACK_LENGTH  # Backward propagation
        time_elapsed = t2 - t1
        
        if time_elapsed > 0:
            return distance / time_elapsed  # m/s
        return None

    def measure_throughput(self):
        """Count vehicles that crossed reference point this second."""
        crossed = 0
        for i in range(NUM_CARS):
            old_pos = self.last_positions[i]
            new_pos = self.positions[i]
            
            # Check if vehicle crossed the reference point
            if old_pos <= self.reference_point < new_pos:
                crossed += 1
            elif old_pos > new_pos:  # Wrapped around
                if old_pos <= self.reference_point or new_pos >= self.reference_point:
                    crossed += 1
            
            self.last_positions[i] = new_pos
        
        self.vehicles_passed_total += crossed
        return crossed

    def calculate_speed_variance_by_type(self):
        """Calculate speed variance separately for cars and trucks."""
        car_speeds = [self.velocities[i] for i in range(NUM_CARS) 
                      if self.vehicle_types[i] == "car"]
        truck_speeds = [self.velocities[i] for i in range(NUM_CARS) 
                        if self.vehicle_types[i] == "truck"]
        
        car_variance = np.var(car_speeds) if car_speeds else 0
        truck_variance = np.var(truck_speeds) if truck_speeds else 0
        
        car_mean = np.mean(car_speeds) if car_speeds else 0
        truck_mean = np.mean(truck_speeds) if truck_speeds else 0
        
        return car_mean, car_variance, truck_mean, truck_variance

    def calculate_gap_distribution(self):
        """Calculate gap statistics for all vehicles."""
        gaps = []
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            gaps.append(gap)
        
        return {
            'min': min(gaps),
            'max': max(gaps),
            'mean': np.mean(gaps),
            'std': np.std(gaps),
            'median': np.median(gaps)
        }

    def log_data(self):
        """Record comprehensive statistics for this second."""
        # Basic velocity stats
        avg_velocity = sum(self.velocities) / NUM_CARS
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        velocity_variance = np.var(self.velocities)
        
        # Throughput
        throughput_this_second = self.measure_throughput()
        
        # Vehicle state counts
        cruising = sum(1 for i in range(NUM_CARS) 
                      if self.velocities[i] >= self.v_desired[i] - 1.0)
        braking = sum(1 for i in range(NUM_CARS) if self.accelerations[i] < -1.0)
        stopped = sum(1 for v in self.velocities if v < 1.0)
        accelerating = sum(1 for i in range(NUM_CARS) if self.accelerations[i] > 0.5)
        
        # Fleet composition
        num_cars = self.vehicle_types.count("car")
        num_trucks = NUM_CARS - num_cars
        num_aggressive = self.personalities.count("aggressive")
        num_cautious = NUM_CARS - num_aggressive
        
        # Gap distribution
        gap_stats = self.calculate_gap_distribution()
        
        # Speed variance by type
        car_mean, car_var, truck_mean, truck_var = self.calculate_speed_variance_by_type()
        
        # Acceleration stats
        avg_accel = np.mean(self.accelerations)
        max_accel = max(self.accelerations)
        min_accel = min(self.accelerations)
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado_sim": self.estado,
            
            # Fleet composition
            "num_cars": num_cars,
            "num_trucks": num_trucks,
            "num_aggressive": num_aggressive,
            "num_cautious": num_cautious,
            
            # Velocity metrics
            "avg_velocity": round(avg_velocity, 3),
            "min_velocity": round(min_velocity, 3),
            "max_velocity": round(max_velocity, 3),
            "velocity_variance": round(velocity_variance, 3),
            
            # Velocity by vehicle type
            "car_mean_velocity": round(car_mean, 3),
            "car_velocity_variance": round(car_var, 3),
            "truck_mean_velocity": round(truck_mean, 3),
            "truck_velocity_variance": round(truck_var, 3),
            
            # Throughput
            "throughput_per_second": throughput_this_second,
            "cumulative_throughput": self.vehicles_passed_total,
            "throughput_per_min": throughput_this_second * 60,
            
            # Vehicle states
            "vehicles_cruising": cruising,
            "vehicles_braking": braking,
            "vehicles_stopped": stopped,
            "vehicles_accelerating": accelerating,
            
            # Gap distribution
            "min_gap": round(gap_stats['min'], 2),
            "max_gap": round(gap_stats['max'], 2),
            "avg_gap": round(gap_stats['mean'], 2),
            "std_gap": round(gap_stats['std'], 2),
            "median_gap": round(gap_stats['median'], 2),
            
            # Safety
            "near_collisions": self.near_collisions_count,
            "hard_brakes": self.hard_brakes_count,
            
            # Acceleration
            "avg_acceleration": round(avg_accel, 3),
            "max_acceleration": round(max_accel, 3),
            "min_acceleration": round(min_accel, 3),
        })

    def get_vehicle_color(self, i):
        """Calculate vehicle color based on acceleration state."""
        accel = self.accelerations[i]
        
        # Normalize acceleration to [-1, 1]
        norm_accel = max(-1.0, min(1.0, accel / 3.0))
        
        if norm_accel > 0:
            # Accelerating: yellow to green
            intensity = norm_accel
            r = int(255 * (1 - intensity))
            g = 255
            b = 0
        else:
            # Braking: yellow to red
            intensity = abs(norm_accel)
            r = 255
            g = int(255 * (1 - intensity))
            b = 0
        
        return (r, g, b)

    def draw(self):
        """Render simulation visualization."""
        if not self.enable_visual:
            return
        
        self.screen.fill((20, 20, 25))
        
        # Draw track
        center_x, center_y = 600, 475
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), 
                          TRACK_RADIUS, 30)
        
        # Reference line for throughput measurement
        ref_angle = (self.reference_point / TRACK_LENGTH) * 2 * math.pi
        ref_x1 = center_x + (TRACK_RADIUS - 50) * math.cos(ref_angle)
        ref_y1 = center_y + (TRACK_RADIUS - 50) * math.sin(ref_angle)
        ref_x2 = center_x + (TRACK_RADIUS + 50) * math.cos(ref_angle)
        ref_y2 = center_y + (TRACK_RADIUS + 50) * math.sin(ref_angle)
        pygame.draw.line(self.screen, (100, 255, 100), 
                        (int(ref_x1), int(ref_y1)), (int(ref_x2), int(ref_y2)), 2)
        
        # Draw vehicles
        for i in range(NUM_CARS):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_vehicle_color(i)
            
            # Size based on vehicle type
            if self.vehicle_types[i] == "car":
                radius = 8
            else:  # truck
                radius = 13
            
            # White outline for disturbance source
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), radius + 2)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        # Statistics panel
        avg_vel = sum(self.velocities) / NUM_CARS
        stopped = sum(1 for v in self.velocities if v < 1.0)
        car_mean, _, truck_mean, _ = self.calculate_speed_variance_by_type()
        
        info = [
            f"SCENARIO: {self.config['nombre']}",
            f"Trucks: {int(self.config['truck_pct']*100)}%  "
            f"Aggressive: {int(self.config['aggressive_pct']*100)}%",
            "",
            f"TIME: {self.segundo_actual}s   STATE: {self.estado}",
            "",
            "=== FLEET ===",
            f"Cars:       {self.vehicle_types.count('car')}  "
            f"(avg: {car_mean:.1f} m/s)",
            f"Trucks:     {self.vehicle_types.count('truck')}  "
            f"(avg: {truck_mean:.1f} m/s)",
            f"Aggressive: {self.personalities.count('aggressive')}",
            f"Cautious:   {self.personalities.count('cautious')}",
            "",
            "=== FLOW ===",
            f"Avg velocity:  {avg_vel:.2f} m/s",
            f"Min velocity:  {min(self.velocities):.2f} m/s",
            f"Max velocity:  {max(self.velocities):.2f} m/s",
            "",
            "=== THROUGHPUT ===",
            f"Total passed:  {self.vehicles_passed_total}",
            f"Rate: {self.vehicles_passed_total * 60 // max(1, self.segundo_actual)} veh/min",
            "",
            "=== STATES ===",
            f"Cruising:      {sum(1 for i in range(NUM_CARS) if self.velocities[i] >= self.v_desired[i] - 1.0)}",
            f"Braking:       {sum(1 for a in self.accelerations if a < -1.0)}",
            f"Stopped:       {stopped}",
            f"Accelerating:  {sum(1 for a in self.accelerations if a > 0.5)}",
            "",
            "=== SAFETY ===",
            f"Near collisions: {self.near_collisions_count}",
            f"Hard brakes:     {self.hard_brakes_count}",
            f"Max stopped:     {self.max_stopped}",
        ]
        
        for idx, text in enumerate(info):
            if text == "":
                continue
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 15 + idx * 18))
        
        # Legend
        legend_x, legend_y = 900, 800
        legend_items = [
            ((255, 0, 0), "Hard braking", 8),
            ((255, 255, 0), "Coasting", 8),
            ((127, 255, 0), "Accelerating", 8),
            ((150, 150, 150), "Car", 8),
            ((150, 150, 150), "Truck", 13),
        ]
        
        for i, (color, label, size) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            pygame.draw.circle(self.screen, color, (legend_x, y_pos), size)
            self.screen.blit(self.font.render(label, True, (200, 200, 200)),
                            (legend_x + 20, y_pos - 8))
        
        pygame.display.flip()

    def save_csv(self):
        """Save time-series data to CSV."""
        os.makedirs("resultados_heterogeneo_v2", exist_ok=True)
        filename = f"resultados_heterogeneo_v2/{self.config['nombre']}_datos.csv"
        
        if not self.logs:
            print(f"  ⚠️  No data to save")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.logs[0].keys())
            writer.writeheader()
            writer.writerows(self.logs)
        
        print(f"  ✅  {filename}")

    def run(self):
        """Main simulation loop."""
        running = True
        
        while running and self.segundo_actual < MAX_TIME:
            # Handle pygame events
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # State transitions
            if self.estado == "ESTABILIZANDO" and self.segundo_actual >= DISTURBANCE_START:
                self.estado = "FRENANDO"
                self.timer_frenada = DISTURBANCE_DURATION
            
            if self.estado == "FRENANDO":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "RECUPERANDO"
            
            # Physics update
            self.step_physics()
            self.update_metrics()
            
            # Data logging (every second)
            self.frame_count += 1
            if self.frame_count >= 10:  # 10 frames * 0.1s = 1 second
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                # Check for jam dissolution
                if self.estado == "RECUPERANDO":
                    all_at_target = all(self.velocities[i] >= self.v_desired[i] - 1.0 
                                       for i in range(NUM_CARS))
                    if all_at_target and self.t_dissolve is None:
                        self.t_dissolve = self.segundo_actual
                        self.wave_speed = self.calculate_wave_speed()
                        self.estado = "FIN_ESPERA"
                        self.timer_fin = 5
                
                # Wait before finishing
                if self.estado == "FIN_ESPERA":
                    self.timer_fin -= 1
                    if self.timer_fin <= 0:
                        running = False
                
                # Perpetual jam detection (after 200 seconds)
                if self.segundo_actual > 200 and self.estado == "RECUPERANDO":
                    stopped = sum(1 for v in self.velocities if v < 1.0)
                    if stopped > NUM_CARS * 0.3:  # More than 30% still stopped
                        print(f"  ⚠️  Perpetual jam detected")
                        self.jam_perpetuo = True
                        running = False
            
            # Visualization
            if self.enable_visual:
                self.draw()
                self.clock.tick(VISUAL_SPEED)
        
        # Save results
        self.save_csv()
        
        # Prepare summary
        summary = {
            "nombre": self.config["nombre"],
            "truck_pct": self.config["truck_pct"],
            "aggressive_pct": self.config["aggressive_pct"],
            "t_dissolve": self.t_dissolve if not self.jam_perpetuo else "NaN",
            "wave_speed": round(self.wave_speed, 3) if self.wave_speed else "NaN",
            "max_stopped": self.max_stopped,
            "final_throughput": self.vehicles_passed_total,
            "near_collisions": self.near_collisions_count,
            "hard_brakes": self.hard_brakes_count,
            "jam_perpetuo": int(self.jam_perpetuo),
        }
        
        return "NEXT", summary


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC SIMULATION V2")
    print("=" * 70)
    print(f"  Scenarios: {len(ESCENARIOS)}")
    print(f"  Vehicles: {NUM_CARS}")
    print(f"  Track: {TRACK_LENGTH:.0f} m")
    print(f"  Visualization: {'ENABLED' if ENABLE_VISUAL else 'DISABLED'}")
    print("=" * 70 + "\n")
    
    master_results = []
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i:02d}/{len(ESCENARIOS)}] {config['nombre']} ", end="", flush=True)
        
        sim = SimuladorHeterogeneo(config, enable_visual=ENABLE_VISUAL)
        result, summary = sim.run()
        
        master_results.append(summary)
        
        if result == "QUIT":
            print("\n⛔  Interrupted by user")
            break
        
        print(f"✓ T_dissolve={summary['t_dissolve']}s  "
              f"Throughput={summary['final_throughput']}")
    
    # Save master results
    if master_results:
        with open("resultados_heterogeneo_v2_master.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_results[0].keys())
            writer.writeheader()
            writer.writerows(master_results)
        print(f"\n✅  Master results: resultados_heterogeneo_v2_master.csv")
    
    if ENABLE_VISUAL:
        pygame.quit()
    
    print("\n✅  Simulation complete!\n")
