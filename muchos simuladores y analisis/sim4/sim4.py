"""
simulador_heterogeneo.py — TFG Heterogeneous Traffic Flow
===========================================================
Based on the original homogeneous simulation, now with:
    - Two vehicle types: CARS (fast) and TRUCKS (slow, larger)
    - Binary driver personality: AGGRESSIVE or CAUTIOUS (not 0-1 scale)
    - Visual: color = acceleration, size = vehicle type
    - Comprehensive data logging with throughput measurement

EXPERIMENT GRID:
    Truck %: 0%, 5%, 10%, 25%, 50%, 75%
    Aggressive %: 0%, 25%, 50%, 75%, 100%
    Total: 6 × 5 = 30 scenarios

Uso:
    python simulador_heterogeneo.py
"""

import pygame
import math
import csv
import random
import os

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
#  PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════
NUM_CARS = 38                        # Total vehicles on track
TRACK_RADIUS = 400                   # meters
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS

# Vehicle specifications
CAR_LENGTH = 5.0                     # meters
TRUCK_LENGTH = 12.0                  # meters
CAR_MAX_SPEED = 30.0                 # m/s
TRUCK_MAX_SPEED = 25.0               # m/s

# IDM parameters (from original simulation)
S0 = 5.0                            # minimum gap (meters)
ACC_MAX = 1.2                       # maximum acceleration (m/s²)
DEC_MAX = 2.0                       # maximum deceleration (m/s²)

# Driver personality parameters (BINARY: aggressive or cautious)
T_REACCION_AGGRESSIVE = 1.0         # seconds
T_REACCION_CAUTIOUS = 2.0           # seconds

# Disturbance parameters
DISTURBANCE_START = 7.0             # seconds
DISTURBANCE_DURATION = 5.0          # seconds
DISTURBANCE_DECEL = -12.0           # m/s²

# Simulation parameters
MAX_TIME = 400                      # seconds (cutoff for perpetual jams)
DT = 0.1                           # timestep


# ═══════════════════════════════════════════════════════════════════
#  SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════
class SimuladorHeterogeneo:
    def __init__(self, config):
        """Initialize simulation with fleet composition from config."""
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((1100, 950))
        pygame.display.set_caption(f"TFG — {config['nombre']}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        
        self.config = config
        
        # ── Create vehicle fleet ──────────────────────────────────
        # Determine how many of each type
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
        
        # Initialize vehicle properties
        self.vehicle_types = vehicle_types
        self.personalities = personalities
        self.lengths = []
        self.max_speeds = []
        self.t_reacciones = []
        
        for vtype, pers in zip(vehicle_types, personalities):
            # Set length and max speed based on type
            if vtype == "car":
                self.lengths.append(CAR_LENGTH)
                self.max_speeds.append(CAR_MAX_SPEED)
            else:  # truck
                self.lengths.append(TRUCK_LENGTH)
                self.max_speeds.append(TRUCK_MAX_SPEED)
            
            # Set reaction time based on personality
            if pers == "aggressive":
                self.t_reacciones.append(T_REACCION_AGGRESSIVE)
            else:  # cautious
                self.t_reacciones.append(T_REACCION_CAUTIOUS)
        
        # ── State variables ───────────────────────────────────────
        spacing = TRACK_LENGTH / NUM_CARS
        self.positions = [i * spacing for i in range(NUM_CARS)]
        self.velocities = [0.0] * NUM_CARS
        self.accelerations = [0.0] * NUM_CARS
        
        # Target speeds (personality affects this)
        self.v_targets = []
        for i in range(NUM_CARS):
            if self.personalities[i] == "aggressive":
                self.v_targets.append(self.max_speeds[i])
            else:
                self.v_targets.append(self.max_speeds[i] * 0.9)
        
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

    def calculate_idm_acceleration(self, i):
        """Calculate IDM acceleration for vehicle i."""
        lead = (i + 1) % NUM_CARS
        
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - self.lengths[lead])
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        v_target = self.v_targets[i]
        t_react = self.t_reacciones[i]
        
        if v_target > 0:
            v_ratio = v / v_target
        else:
            v_ratio = 0
        free_term = ACC_MAX * (1.0 - v_ratio ** 4)
        
        s_star = S0 + max(0, v * t_react + 
                         (v * dv) / (2 * math.sqrt(ACC_MAX * DEC_MAX)))
        interaction = ACC_MAX * (s_star / gap) ** 2
        
        accel = free_term - interaction
        return accel

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        if self.estado == "ESTABILIZANDO":
            for i in range(NUM_CARS):
                if self.velocities[i] < self.v_targets[i]:
                    self.accelerations[i] = ACC_MAX * 0.3
                else:
                    self.accelerations[i] = 0.0
        else:
            for i in range(NUM_CARS):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        if self.estado == "FRENANDO":
            self.accelerations[0] = DISTURBANCE_DECEL
        
        for i in range(NUM_CARS):
            self.velocities[i] += self.accelerations[i] * DT
            self.velocities[i] = max(0.0, min(self.max_speeds[i], self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def update_metrics(self):
        """Calculate real-time safety metrics."""
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            
            if gap < 2.0 and self.velocities[i] > 2.0:
                self.near_collisions_count += 1
            
            if self.accelerations[i] < -4.0:
                self.hard_brakes_count += 1
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        if stopped > self.max_stopped:
            self.max_stopped = stopped

    def measure_throughput(self):
        """Count vehicles that crossed reference point this second."""
        crossed = 0
        for i in range(NUM_CARS):
            old_pos = self.last_positions[i]
            new_pos = self.positions[i]
            
            if old_pos <= self.reference_point < new_pos:
                crossed += 1
            elif old_pos > new_pos:
                if old_pos <= self.reference_point or new_pos >= self.reference_point:
                    crossed += 1
            
            self.last_positions[i] = new_pos
        
        self.vehicles_passed_total += crossed
        return crossed

    def log_data(self):
        """Record comprehensive statistics for this second."""
        avg_velocity = sum(self.velocities) / NUM_CARS
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        velocity_variance = sum((v - avg_velocity)**2 for v in self.velocities) / NUM_CARS
        
        throughput_this_second = self.measure_throughput()
        
        cruising = sum(1 for i in range(NUM_CARS) if self.velocities[i] >= self.v_targets[i] - 1.0)
        braking = sum(1 for i in range(NUM_CARS) if self.accelerations[i] < -1.0)
        stopped = sum(1 for v in self.velocities if v < 1.0)
        accelerating = sum(1 for i in range(NUM_CARS) if self.accelerations[i] > 0.5)
        
        num_cars = self.vehicle_types.count("car")
        num_trucks = NUM_CARS - num_cars
        num_aggressive = self.personalities.count("aggressive")
        num_cautious = NUM_CARS - num_aggressive
        
        gaps = []
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            gaps.append(gap)
        
        min_gap = min(gaps)
        avg_gap = sum(gaps) / len(gaps)
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado_sim": self.estado,
            "num_cars": num_cars,
            "num_trucks": num_trucks,
            "num_aggressive": num_aggressive,
            "num_cautious": num_cautious,
            "avg_velocity": round(avg_velocity, 3),
            "min_velocity": round(min_velocity, 3),
            "max_velocity": round(max_velocity, 3),
            "velocity_variance": round(velocity_variance, 3),
            "throughput_per_second": throughput_this_second,
            "cumulative_throughput": self.vehicles_passed_total,
            "throughput_per_min": throughput_this_second * 60,
            "vehicles_cruising": cruising,
            "vehicles_braking": braking,
            "vehicles_stopped": stopped,
            "vehicles_accelerating": accelerating,
            "min_gap": round(min_gap, 2),
            "avg_gap": round(avg_gap, 2),
            "near_collisions": self.near_collisions_count,
            "hard_brakes": self.hard_brakes_count,
            "avg_acceleration": round(sum(self.accelerations) / NUM_CARS, 3),
            "max_acceleration": round(max(self.accelerations), 3),
            "min_acceleration": round(min(self.accelerations), 3),
        })

    def get_vehicle_color(self, i):
        """
        Calculate vehicle color based on acceleration.
        Green = accelerating, Yellow = coasting, Red = braking
        """
        accel = self.accelerations[i]
        norm_accel = max(-1.0, min(1.0, accel / 3.0))
        
        if norm_accel > 0:
            intensity = norm_accel
            r = int(255 * intensity)
            g = 255
            b = 0
        else:
            intensity = abs(norm_accel)
            r = 255
            g = int(255 * (1 - intensity))
            b = 0
        
        return (r, g, b)

    def draw(self):
        """Render simulation with color-coded vehicles and stats panel."""
        self.screen.fill((20, 20, 25))
        
        center_x, center_y = 550, 450
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), 
                          TRACK_RADIUS, 30)
        
        # Reference line
        ref_angle = (self.reference_point / TRACK_LENGTH) * 2 * math.pi
        ref_x1 = center_x + (TRACK_RADIUS - 50) * math.cos(ref_angle)
        ref_y1 = center_y + (TRACK_RADIUS - 50) * math.sin(ref_angle)
        ref_x2 = center_x + (TRACK_RADIUS + 50) * math.cos(ref_angle)
        ref_y2 = center_y + (TRACK_RADIUS + 50) * math.sin(ref_angle)
        pygame.draw.line(self.screen, (100, 255, 100), 
                        (int(ref_x1), int(ref_y1)), (int(ref_x2), int(ref_y2)), 2)
        
        # Vehicles
        for i in range(NUM_CARS):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_vehicle_color(i)
            
            # Size based on vehicle type
            radius = 8 if self.vehicle_types[i] == "car" else 13
            
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 
                                 radius + 2)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        # Statistics panel
        avg_vel = sum(self.velocities) / NUM_CARS
        stopped = sum(1 for v in self.velocities if v < 1.0)
        
        info = [
            f"SCENARIO: {self.config['nombre']}",
            f"Trucks: {int(self.config['truck_pct']*100)}%  "
            f"Aggressive: {int(self.config['aggressive_pct']*100)}%",
            f"",
            f"TIME: {self.segundo_actual}s   STATE: {self.estado}",
            f"",
            f"=== FLEET COMPOSITION ===",
            f"Cars:       {self.vehicle_types.count('car')}",
            f"Trucks:     {self.vehicle_types.count('truck')}",
            f"Aggressive: {self.personalities.count('aggressive')}",
            f"Cautious:   {self.personalities.count('cautious')}",
            f"",
            f"=== FLOW METRICS ===",
            f"Avg velocity:  {avg_vel:.2f} m/s",
            f"Min velocity:  {min(self.velocities):.2f} m/s",
            f"Max velocity:  {max(self.velocities):.2f} m/s",
            f"",
            f"=== THROUGHPUT ===",
            f"Total passed:  {self.vehicles_passed_total}",
            f"Rate:          {self.vehicles_passed_total * 60 // max(1, self.segundo_actual)} veh/min",
            f"",
            f"=== VEHICLE STATES ===",
            f"Cruising:      {sum(1 for i in range(NUM_CARS) if self.velocities[i] >= self.v_targets[i] - 1.0)}",
            f"Braking:       {sum(1 for a in self.accelerations if a < -1.0)}",
            f"Stopped:       {stopped}",
            f"Accelerating:  {sum(1 for a in self.accelerations if a > 0.5)}",
            f"",
            f"=== SAFETY ===",
            f"Near collisions: {self.near_collisions_count}",
            f"Hard brakes:     {self.hard_brakes_count}",
            f"Max stopped:     {self.max_stopped}",
            f"",
            f"[Circle size = vehicle type]",
            f"[Color = acceleration]",
            f"[White outline = disturbance source]",
        ]
        
        for idx, text in enumerate(info):
            if text == "":
                continue
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 15 + idx * 20))
        
        # Color legend
        legend_x, legend_y = 850, 820
        pygame.draw.circle(self.screen, (255, 0, 0), (legend_x, legend_y), 8)
        self.screen.blit(self.font.render("Hard braking", True, (200, 200, 200)),
                        (legend_x + 15, legend_y - 8))
        pygame.draw.circle(self.screen, (255, 255, 0), (legend_x, legend_y + 25), 8)
        self.screen.blit(self.font.render("Coasting", True, (200, 200, 200)),
                        (legend_x + 15, legend_y + 17))
        pygame.draw.circle(self.screen, (127, 255, 0), (legend_x, legend_y + 50), 8)
        self.screen.blit(self.font.render("Accelerating", True, (200, 200, 200)),
                        (legend_x + 15, legend_y + 42))
        pygame.draw.circle(self.screen, (150, 150, 150), (legend_x, legend_y + 80), 8)
        self.screen.blit(self.font.render("Car", True, (200, 200, 200)),
                        (legend_x + 15, legend_y + 72))
        pygame.draw.circle(self.screen, (150, 150, 150), (legend_x, legend_y + 105), 13)
        self.screen.blit(self.font.render("Truck", True, (200, 200, 200)),
                        (legend_x + 15, legend_y + 97))
        
        pygame.display.flip()

    def save_csv(self):
        """Save time-series data to CSV."""
        os.makedirs("resultados_heterogeneo", exist_ok=True)
        filename = f"resultados_heterogeneo/{self.config['nombre']}_datos.csv"
        
        if not self.logs:
            print(f"  ⚠️  No data")
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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
            
            if self.estado == "ESTABILIZANDO" and self.segundo_actual >= DISTURBANCE_START:
                self.estado = "FRENANDO"
                self.timer_frenada = DISTURBANCE_DURATION
            
            if self.estado == "FRENANDO":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "RECUPERANDO"
            
            self.step_physics()
            self.update_metrics()
            
            self.frame_count += 1
            if self.frame_count >= 10:
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                if self.estado == "RECUPERANDO":
                    all_at_target = all(self.velocities[i] >= self.v_targets[i] - 1.0 
                                       for i in range(NUM_CARS))
                    if all_at_target and self.t_dissolve is None:
                        self.t_dissolve = self.segundo_actual
                        self.estado = "FIN_ESPERA"
                        self.timer_fin = 5
                
                if self.estado == "FIN_ESPERA":
                    self.timer_fin -= 1
                    if self.timer_fin <= 0:
                        running = False
                
                if self.segundo_actual > 350 and self.estado == "RECUPERANDO":
                    print(f"  ⚠️  Perpetual jam")
                    self.jam_perpetuo = True
                    running = False
            
            self.draw()
            self.clock.tick(30)
        
        self.save_csv()
        
        return "NEXT", {
            "nombre": self.config["nombre"],
            "truck_pct": self.config["truck_pct"],
            "aggressive_pct": self.config["aggressive_pct"],
            "t_dissolve": self.t_dissolve if not self.jam_perpetuo else "NaN",
            "max_stopped": self.max_stopped,
            "final_throughput": self.vehicles_passed_total,
            "near_collisions": self.near_collisions_count,
            "hard_brakes": self.hard_brakes_count,
            "jam_perpetuo": int(self.jam_perpetuo),
        }


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC SIMULATION")
    print("=" * 70)
    print(f"  Scenarios: {len(ESCENARIOS)}")
    print(f"  Vehicles: {NUM_CARS}")
    print(f"  Track: {TRACK_LENGTH:.0f} m")
    print("=" * 70 + "\n")
    
    master_results = []
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i:02d}/{len(ESCENARIOS)}] {config['nombre']} ", end="")
        
        sim = SimuladorHeterogeneo(config)
        result, summary = sim.run()
        
        master_results.append(summary)
        
        if result == "QUIT":
            print("\n⛔  Interrupted")
            break
        
        print(f"T_dissolve={summary['t_dissolve']}s  Throughput={summary['final_throughput']}")
    
    if master_results:
        with open("resultados_heterogeneo_master.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_results[0].keys())
            writer.writeheader()
            writer.writerows(master_results)
        print(f"\n✅  Master: resultados_heterogeneo_master.csv")
    
    pygame.quit()
    print("\n✅  Complete!\n")
