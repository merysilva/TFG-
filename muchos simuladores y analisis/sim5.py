"""
simulador_heterogeneo_v2_FIXED.py — Improved Heterogeneous Traffic Flow Simulation
===================================================================================
Enhanced version with:
    - Realistic vehicle differentiation (trucks vs cars)
    - Binary driver personality (aggressive vs cautious)
    - Comprehensive data logging with wave propagation metrics
    - No permanent jams through calibrated parameters
    - Optional visualization (ENABLE_VISUAL flag)
    - ✅ FIXED: Proper jam dissolution detection using velocity variance
    - ✅ FIXED: Enhanced 6-state machine with bidirectional transitions
    - ✅ FIXED: Realistic stop behavior
    - ✅ FIXED: v_diff metric tracking

Uso:
    python simulador_heterogeneo_v2_FIXED.py
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
        self.sub_estado = None  # ✅ FIXED: Track recovery sub-phases
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
        
        if gap > 0:
            interaction = acc_max * (s_desired / gap) ** 2
        else:
            interaction = acc_max * 100  # Emergency brake
        
        accel = free_term - interaction
        
        # Limit to vehicle capabilities
        accel = max(-dec_max, min(acc_max, accel))
        
        return accel

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        # Calculate accelerations based on state
        if self.estado == "ESTABILIZANDO":
            # Gentle warm-up acceleration
            for i in range(NUM_CARS):
                if self.velocities[i] < self.v_desired[i]:
                    self.accelerations[i] = self.acc_maxes[i] * 0.5
                else:
                    self.accelerations[i] = 0.0
        
        elif self.estado == "FLUJO_SINCRO":
            # Use IDM for synchronized flow
            for i in range(NUM_CARS):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        else:
            # Normal IDM control for all other states
            for i in range(NUM_CARS):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance
        if self.estado == "FRENADA_ORIGEN":
            self.accelerations[0] = DISTURBANCE_DECEL
        
        # Update velocities and positions
        for i in range(NUM_CARS):
            self.velocities[i] += self.accelerations[i] * DT
            
            # ✅ FIXED: Realistic stop behavior
            # If moving very slowly and still braking, force a complete stop
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
                self.accelerations[i] = 0.0
            
            self.velocities[i] = max(0.0, min(self.max_speeds[i], self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def update_metrics(self):
        """Update various metrics for analysis."""
        # Track throughput
        for i in range(NUM_CARS):
            if self.last_positions[i] > self.reference_point >= self.positions[i]:
                self.vehicles_passed_total += 1
        self.last_positions = self.positions.copy()
        
        # Calculate gaps and check for near collisions
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            
            # Near collision: gap < 2m while moving > 2 m/s
            if gap < 2.0 and self.velocities[i] > 2.0:
                self.near_collisions_count += 1
        
        # Hard braking events
        for a in self.accelerations:
            if a < -4.0:  # Hard brake threshold
                self.hard_brakes_count += 1
        
        # Track maximum stopped vehicles
        stopped = sum(1 for v in self.velocities if v < 1.0)
        if stopped > self.max_stopped:
            self.max_stopped = stopped

    def calculate_wave_speed(self):
        """Calculate the jam wave propagation speed."""
        if not self.wave_positions or len(self.wave_positions) < 2:
            return None
        
        # Use first and last position to calculate average wave speed
        time_diff = self.wave_positions[-1][0] - self.wave_positions[0][0]
        if time_diff <= 0:
            return None
        
        pos_diff = abs(self.wave_positions[-1][1] - self.wave_positions[0][1])
        wave_speed = pos_diff / time_diff
        
        return wave_speed

    def log_data(self):
        """Record comprehensive statistics for this second."""
        # Basic velocity statistics
        avg_velocity = sum(self.velocities) / NUM_CARS
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        velocity_variance = np.var(self.velocities)
        
        # ✅ FIXED: Add velocity variance metric (v_diff)
        v_diff = max_velocity - min_velocity
        
        # Vehicle type statistics
        car_velocities = [self.velocities[i] for i in range(NUM_CARS) 
                         if self.vehicle_types[i] == "car"]
        truck_velocities = [self.velocities[i] for i in range(NUM_CARS) 
                           if self.vehicle_types[i] == "truck"]
        
        car_mean = sum(car_velocities) / len(car_velocities) if car_velocities else 0
        truck_mean = sum(truck_velocities) / len(truck_velocities) if truck_velocities else 0
        car_variance = np.var(car_velocities) if len(car_velocities) > 1 else 0
        truck_variance = np.var(truck_velocities) if len(truck_velocities) > 1 else 0
        
        # Gap statistics
        gaps = []
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        median_gap = sorted(gaps)[len(gaps) // 2]
        
        # Vehicle state counts
        stopped = sum(1 for v in self.velocities if v < 1.0)
        cruising = sum(1 for i in range(NUM_CARS) 
                      if self.velocities[i] >= self.v_desired[i] - 1.0)
        braking = sum(1 for a in self.accelerations if a < -1.0)
        accelerating = sum(1 for a in self.accelerations if a > 0.5)
        
        # Wave tracking
        if self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
            # Find the position of the jam front (first stopped vehicle)
            for i in range(NUM_CARS):
                if self.velocities[i] < 1.0:
                    self.wave_positions.append((self.segundo_actual, self.positions[i]))
                    break
        
        # Log everything
        log_entry = {
            "segundo": self.segundo_actual,
            "estado": self.estado,
            "sub_estado": self.sub_estado if self.sub_estado else "",
            
            # Velocity metrics
            "avg_velocity": round(avg_velocity, 2),
            "min_velocity": round(min_velocity, 2),
            "max_velocity": round(max_velocity, 2),
            "velocity_variance": round(velocity_variance, 2),
            "v_diff": round(v_diff, 2),  # ✅ FIXED: Added v_diff
            
            # Type-specific metrics
            "car_mean_velocity": round(car_mean, 2),
            "truck_mean_velocity": round(truck_mean, 2),
            "car_velocity_variance": round(car_variance, 2),
            "truck_velocity_variance": round(truck_variance, 2),
            
            # Gap metrics
            "avg_gap": round(avg_gap, 2),
            "min_gap": round(min_gap, 2),
            "median_gap": round(median_gap, 2),
            
            # Vehicle states
            "vehicles_stopped": stopped,
            "vehicles_cruising": cruising,
            "vehicles_braking": braking,
            "vehicles_accelerating": accelerating,
            
            # Throughput and safety
            "cumulative_throughput": self.vehicles_passed_total,
            "near_collisions": self.near_collisions_count,
            "hard_brakes": self.hard_brakes_count,
        }
        
        self.logs.append(log_entry)

    def get_vehicle_color(self, i):
        """Calculate vehicle color based on acceleration."""
        accel = self.accelerations[i]
        
        # Normalize acceleration to [-1, 1]
        max_accel = max(self.acc_maxes[i], self.dec_maxes[i])
        norm_accel = max(-1.0, min(1.0, accel / max_accel))
        
        if norm_accel > 0:
            # Accelerating: yellow to green
            intensity = norm_accel
            r = int(255 * (1 - intensity * 0.5))
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
        
        # Draw vehicles
        for i in range(NUM_CARS):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_vehicle_color(i)
            
            # Size based on vehicle type
            if self.vehicle_types[i] == "truck":
                radius = 13
            else:
                radius = 8
            
            # White outline for disturbance source
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), radius + 2)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        # Statistics panel
        avg_vel = sum(self.velocities) / NUM_CARS
        stopped = sum(1 for v in self.velocities if v < 1.0)
        
        car_velocities = [self.velocities[i] for i in range(NUM_CARS) 
                         if self.vehicle_types[i] == "car"]
        truck_velocities = [self.velocities[i] for i in range(NUM_CARS) 
                           if self.vehicle_types[i] == "truck"]
        car_mean = sum(car_velocities) / len(car_velocities) if car_velocities else 0
        truck_mean = sum(truck_velocities) / len(truck_velocities) if truck_velocities else 0
        
        v_diff = max(self.velocities) - min(self.velocities)
        
        info = [
            f"SCENARIO: {self.config['nombre']}",
            f"Trucks: {int(self.config['truck_pct']*100)}%  "
            f"Aggressive: {int(self.config['aggressive_pct']*100)}%",
            "",
            f"TIME: {self.segundo_actual}s   STATE: {self.estado}",
            f"Sub-state: {self.sub_estado or 'N/A'}",
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
            f"v_diff:        {v_diff:.2f} m/s",
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
        """Main simulation loop with improved state machine."""
        running = True
        
        while running and self.segundo_actual < MAX_TIME:
            # Handle pygame events
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # ✅ FIXED: Enhanced state machine transitions
            v_min = min(self.velocities)
            v_max = max(self.velocities)
            v_diff = v_max - v_min
            
            # 1. ESTABILIZANDO -> FLUJO_SINCRO
            if self.estado == "ESTABILIZANDO":
                if v_diff < 1.0 and self.segundo_actual > 5:
                    self.estado = "FLUJO_SINCRO"
                    if not self.enable_visual:
                        print(f"  [{self.segundo_actual}s] 🟢 Stable synchronized flow achieved")
            
            # 2. FLUJO_SINCRO -> FRENADA_ORIGEN
            if self.estado == "FLUJO_SINCRO" and self.segundo_actual >= DISTURBANCE_START:
                self.estado = "FRENADA_ORIGEN"
                self.timer_frenada = DISTURBANCE_DURATION
                if not self.enable_visual:
                    print(f"  [{self.segundo_actual}s] 🚨 Disturbance started")
            
            # 3. FRENADA_ORIGEN -> ONDA_ACTIVA
            if self.estado == "FRENADA_ORIGEN":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "ONDA_ACTIVA"
                    self.sub_estado = "PROPAGATING"
                    if not self.enable_visual:
                        print(f"  [{self.segundo_actual}s] 💥 Jam wave propagating")
            
            # 4. ONDA_ACTIVA <-> DISOLVIENDO (bidirectional!)
            if self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                if v_min < 1.0:
                    if self.estado == "DISOLVIENDO":
                        if not self.enable_visual:
                            print(f"  [{self.segundo_actual}s] ⚠️  Jam wave re-formed")
                    self.estado = "ONDA_ACTIVA"
                    self.sub_estado = "PROPAGATING"
                elif v_min >= 1.0:
                    if self.estado == "ONDA_ACTIVA":
                        if not self.enable_visual:
                            print(f"  [{self.segundo_actual}s] 📉 Queue cleared, dissolving...")
                    self.estado = "DISOLVIENDO"
                    self.sub_estado = "RECOVERING"
            
            # 5. DISOLVIENDO -> RECUPERADO (jam dissolved!)
            if self.estado == "DISOLVIENDO" and v_diff < 2.0:
                if self.t_dissolve is None:
                    self.t_dissolve = self.segundo_actual
                    self.wave_speed = self.calculate_wave_speed()
                    self.estado = "RECUPERADO"
                    self.sub_estado = "COMPLETE"
                    self.timer_fin = 5
                    if not self.enable_visual:
                        print(f"  [{self.segundo_actual}s] ✅ JAM DISSOLVED! (v_diff={v_diff:.2f} m/s)")
            
            # Wait before finishing
            if self.estado == "RECUPERADO":
                self.timer_fin -= DT
                if self.timer_fin <= 0:
                    running = False
            
            # Physics update
            self.step_physics()
            self.update_metrics()
            
            # Data logging (every second)
            self.frame_count += 1
            if self.frame_count >= 10:  # 10 frames * 0.1s = 1 second
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                # Perpetual jam detection (after 200 seconds)
                if self.segundo_actual > 200 and self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                    stopped = sum(1 for v in self.velocities if v < 1.0)
                    if stopped > NUM_CARS * 0.2:  # More than 20% still stopped
                        if not self.enable_visual:
                            print(f"  [{self.segundo_actual}s] ❌ PERPETUAL JAM DETECTED!")
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
    print("  HETEROGENEOUS TRAFFIC SIMULATION V2 - FIXED VERSION")
    print("=" * 70)
    print(f"  Scenarios: {len(ESCENARIOS)}")
    print(f"  Vehicles: {NUM_CARS}")
    print(f"  Track: {TRACK_LENGTH:.0f} m")
    print(f"  Visualization: {'ENABLED' if ENABLE_VISUAL else 'DISABLED'}")
    print("  ✅ FIXES APPLIED:")
    print("     - Velocity variance (v_diff) tracking")
    print("     - Enhanced 6-state machine with bidirectional transitions")
    print("     - Realistic stop behavior (v < 0.5 -> full stop)")
    print("     - Proper jam dissolution detection")
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