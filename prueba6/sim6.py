"""
simulador_heterogeneo_FIXED.py — Actually Fixed Heterogeneous Traffic
======================================================================
CRITICAL FIX: Don't check for dissolution before disturbance ends!

Usage:
    python simulador_heterogeneo_FIXED.py
"""

import pygame
import math
import csv
import random
import os

# Configuration
ENABLE_VISUAL = False
VISUAL_SPEED = 30

# Experiment configuration
TRUCK_PERCENTAGES = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]
AGGRESSIVE_PERCENTAGES = [0.0, 0.25, 0.50, 0.75, 1.0]

ESCENARIOS = []
for truck_pct in TRUCK_PERCENTAGES:
    for aggr_pct in AGGRESSIVE_PERCENTAGES:
        ESCENARIOS.append({
            "nombre": f"Het_T{int(truck_pct*100):02d}_A{int(aggr_pct*100):03d}",
            "truck_pct": truck_pct,
            "aggressive_pct": aggr_pct
        })

# Physical constants
NUM_CARS = 27
TRACK_RADIUS = 400
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS

CAR_LENGTH = 5.0
TRUCK_LENGTH = 12.0
CAR_MAX_SPEED = 30.0
TRUCK_MAX_SPEED = 25.0
CAR_ACC_MAX = 2.5
TRUCK_ACC_MAX = 1.5
CAR_DEC_MAX = 4.5
TRUCK_DEC_MAX = 3.0

S0 = 2.5
T_REACTION_BASE = 1.0

# Driver personalities
AGGRESSIVE_SPEED_FACTOR = 1.0
AGGRESSIVE_GAP_FACTOR = 0.8
CAUTIOUS_SPEED_FACTOR = 0.9
CAUTIOUS_GAP_FACTOR = 1.2

# Disturbance
DISTURBANCE_START = 10.0
DISTURBANCE_DURATION = 5.0
DISTURBANCE_DECEL = -12.0

# Simulation
MAX_TIME = 300
DT = 0.1


class SimuladorHeterogeneo:
    def __init__(self, config, enable_visual=True):
        self.enable_visual = enable_visual
        
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"TFG — {config['nombre']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 15)
        
        self.config = config
        
        # Create fleet
        num_trucks = int(NUM_CARS * config["truck_pct"])
        num_cars = NUM_CARS - num_trucks
        num_aggressive = int(NUM_CARS * config["aggressive_pct"])
        
        vehicle_types = (["truck"] * num_trucks + ["car"] * num_cars)
        random.shuffle(vehicle_types)
        
        personalities = (["aggressive"] * num_aggressive + 
                        ["cautious"] * (NUM_CARS - num_aggressive))
        random.shuffle(personalities)
        
        # Vehicle properties
        self.vehicle_types = vehicle_types
        self.personalities = personalities
        self.lengths = []
        self.max_speeds = []
        self.acc_maxes = []
        self.dec_maxes = []
        self.v_desired = []
        self.gap_factors = []
        
        for vtype, pers in zip(vehicle_types, personalities):
            if vtype == "car":
                self.lengths.append(CAR_LENGTH)
                self.max_speeds.append(CAR_MAX_SPEED)
                self.acc_maxes.append(CAR_ACC_MAX)
                self.dec_maxes.append(CAR_DEC_MAX)
            else:
                self.lengths.append(TRUCK_LENGTH)
                self.max_speeds.append(TRUCK_MAX_SPEED)
                self.acc_maxes.append(TRUCK_ACC_MAX)
                self.dec_maxes.append(TRUCK_DEC_MAX)
            
            if pers == "aggressive":
                self.v_desired.append(self.max_speeds[-1] * AGGRESSIVE_SPEED_FACTOR)
                self.gap_factors.append(AGGRESSIVE_GAP_FACTOR)
            else:
                self.v_desired.append(self.max_speeds[-1] * CAUTIOUS_SPEED_FACTOR)
                self.gap_factors.append(CAUTIOUS_GAP_FACTOR)
        
        # Initial state
        spacing = TRACK_LENGTH / NUM_CARS
        self.positions = [i * spacing for i in range(NUM_CARS)]
        self.velocities = [0.0] * NUM_CARS
        self.accelerations = [0.0] * NUM_CARS
        
        # Simulation state
        self.segundo_actual = 0
        self.frame_count = 0
        self.disturbance_active = False
        self.disturbance_end_time = 0
        
        # Metrics
        self.logs = []
        self.max_stopped = 0
        self.t_dissolve = None
        self.jam_perpetuo = False
        self.stable_counter = 0
        self.jam_occurred = False  # NEW: Track if a jam actually happened

    def calculate_idm_acceleration(self, i):
        lead = (i + 1) % NUM_CARS
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - self.lengths[lead])
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        v_desired = self.v_desired[i]
        acc_max = self.acc_maxes[i]
        dec_max = self.dec_maxes[i]
        gap_factor = self.gap_factors[i]
        
        # Free flow term (v^2 from working config)
        v_ratio = v / v_desired if v_desired > 0 else 0
        free_term = acc_max * (1.0 - v_ratio ** 2)
        
        # Interaction term
        s_desired = S0 * gap_factor + max(0, v * T_REACTION_BASE + 
                                          (v * dv) / (2 * math.sqrt(acc_max * dec_max)))
        interaction = acc_max * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        accel = max(-dec_max, min(acc_max, accel))
        
        return accel

    def step_physics(self):
        for i in range(NUM_CARS):
            self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        if self.disturbance_active:
            self.accelerations[0] = DISTURBANCE_DECEL
        
        for i in range(NUM_CARS):
            self.velocities[i] += self.accelerations[i] * DT
            
            # Realistic stop
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
            
            self.velocities[i] = max(0.0, min(self.max_speeds[i], self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def log_data(self):
        avg_velocity = sum(self.velocities) / NUM_CARS
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        v_diff = max_velocity - min_velocity
        
        cruising = sum(1 for i in range(NUM_CARS) 
                      if self.velocities[i] >= self.v_desired[i] - 1.0)
        braking = sum(1 for a in self.accelerations if a < -1.0)
        stopped = sum(1 for v in self.velocities if v < 1.0)
        accelerating = sum(1 for a in self.accelerations if a > 0.5)
        
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        # Track if a jam occurred
        if stopped > 0 and self.segundo_actual > DISTURBANCE_START:
            self.jam_occurred = True
        
        num_cars = self.vehicle_types.count("car")
        num_trucks = NUM_CARS - num_cars
        num_aggressive = self.personalities.count("aggressive")
        num_cautious = NUM_CARS - num_aggressive
        
        car_speeds = [self.velocities[i] for i in range(NUM_CARS) 
                     if self.vehicle_types[i] == "car"]
        truck_speeds = [self.velocities[i] for i in range(NUM_CARS) 
                       if self.vehicle_types[i] == "truck"]
        
        car_mean = sum(car_speeds) / len(car_speeds) if car_speeds else 0
        truck_mean = sum(truck_speeds) / len(truck_speeds) if truck_speeds else 0
        
        gaps = []
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - self.lengths[lead]
            gaps.append(gap)
        
        # Determine state
        if self.segundo_actual < DISTURBANCE_START:
            estado = "WARMING_UP"
        elif self.disturbance_active:
            estado = "DISTURBANCE"
        elif stopped > 0:
            estado = "JAM_ACTIVE"
        else:
            estado = "RECOVERING"
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado": estado,
            "num_cars": num_cars,
            "num_trucks": num_trucks,
            "num_aggressive": num_aggressive,
            "num_cautious": num_cautious,
            "avg_velocity": round(avg_velocity, 2),
            "min_velocity": round(min_velocity, 2),
            "max_velocity": round(max_velocity, 2),
            "v_diff": round(v_diff, 2),
            "car_mean_velocity": round(car_mean, 2),
            "truck_mean_velocity": round(truck_mean, 2),
            "vehicles_cruising": cruising,
            "vehicles_braking": braking,
            "vehicles_stopped": stopped,
            "vehicles_accelerating": accelerating,
            "min_gap": round(min(gaps), 2),
            "avg_gap": round(sum(gaps) / len(gaps), 2),
            "max_gap": round(max(gaps), 2),
        })

    def get_vehicle_color(self, i):
        accel = self.accelerations[i]
        norm_accel = max(-1.0, min(1.0, accel / 3.0))
        
        if norm_accel > 0:
            intensity = norm_accel
            r = int(255 * (1 - intensity))
            g = 255
            b = 0
        else:
            intensity = abs(norm_accel)
            r = 255
            g = int(255 * (1 - intensity))
            b = 0
        
        return (r, g, b)

    def draw(self):
        if not self.enable_visual:
            return
        
        self.screen.fill((20, 20, 25))
        
        center_x, center_y = 600, 475
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), 
                          TRACK_RADIUS, 30)
        
        for i in range(NUM_CARS):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_vehicle_color(i)
            radius = 8 if self.vehicle_types[i] == "car" else 13
            
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), radius + 2)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        avg_vel = sum(self.velocities) / NUM_CARS
        
        info = [
            f"SCENARIO: {self.config['nombre']}",
            f"Time: {self.segundo_actual}s",
            f"Avg vel: {avg_vel:.2f} m/s",
            f"Stopped: {stopped} / Max: {self.max_stopped}",
        ]
        
        if self.t_dissolve:
            info.append(f"✅ Dissolved: {self.t_dissolve}s")
        elif self.jam_perpetuo:
            info.append("❌ PERPETUAL JAM")
        
        for idx, text in enumerate(info):
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 15 + idx * 20))
        
        pygame.display.flip()

    def save_csv(self):
        os.makedirs("resultados_heterogeneo_final", exist_ok=True)
        filename = f"resultados_heterogeneo_final/{self.config['nombre']}_datos.csv"
        
        if not self.logs:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.logs[0].keys())
            writer.writeheader()
            writer.writerows(self.logs)

    def run(self):
        running = True
        
        while running and self.segundo_actual < MAX_TIME:
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # Physics update
            self.step_physics()
            
            # Data logging every second
            self.frame_count += 1
            if self.frame_count >= 10:
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                # Disturbance control
                if self.segundo_actual == DISTURBANCE_START:
                    self.disturbance_active = True
                    self.disturbance_end_time = self.segundo_actual + DISTURBANCE_DURATION
                
                if self.segundo_actual == self.disturbance_end_time:
                    self.disturbance_active = False
                
                # ONLY check for dissolution AFTER disturbance has ended AND a jam occurred
                if self.segundo_actual > self.disturbance_end_time and self.jam_occurred:
                    stopped = sum(1 for v in self.velocities if v < 1.0)
                    
                    if stopped == 0:
                        self.stable_counter += 1
                    else:
                        self.stable_counter = 0
                    
                    # Jam dissolved: 5 consecutive seconds with no stopped vehicles
                    if self.stable_counter >= 5 and self.t_dissolve is None:
                        self.t_dissolve = self.segundo_actual - 5
                        running = False
                
                # Perpetual jam detection (ONLY after disturbance + some recovery time)
                if self.segundo_actual > DISTURBANCE_START + 180:
                    stopped = sum(1 for v in self.velocities if v < 1.0)
                    if stopped > NUM_CARS * 0.10:
                        self.jam_perpetuo = True
                        running = False
            
            if self.enable_visual:
                self.draw()
                self.clock.tick(VISUAL_SPEED)
        
        # Final check: if we exited loop without dissolving
        if self.t_dissolve is None and self.jam_occurred:
            self.jam_perpetuo = True
        
        self.save_csv()
        
        summary = {
            "nombre": self.config["nombre"],
            "truck_pct": self.config["truck_pct"],
            "aggressive_pct": self.config["aggressive_pct"],
            "t_dissolve": self.t_dissolve if not self.jam_perpetuo else "NaN",
            "max_stopped": self.max_stopped,
            "jam_perpetuo": int(self.jam_perpetuo),
        }
        
        return "NEXT", summary


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS TRAFFIC SIMULATION (ACTUALLY FIXED)")
    print("=" * 70)
    print(f"  Scenarios: {len(ESCENARIOS)}")
    print(f"  Visualization: {'ENABLED' if ENABLE_VISUAL else 'DISABLED'}")
    print("=" * 70 + "\n")
    
    master_results = []
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i:02d}/{len(ESCENARIOS)}] {config['nombre']} ", end="", flush=True)
        
        sim = SimuladorHeterogeneo(config, enable_visual=ENABLE_VISUAL)
        result, summary = sim.run()
        
        master_results.append(summary)
        
        if result == "QUIT":
            print("\n⛔  Interrupted")
            break
        
        print(f"✓ T_dissolve={summary['t_dissolve']}s  Max_stopped={summary['max_stopped']}")
    
    if master_results:
        with open("resultados_heterogeneo_final_master.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_results[0].keys())
            writer.writeheader()
            writer.writerows(master_results)
        print(f"\n✅  Master: resultados_heterogeneo_final_master.csv")
    
    if ENABLE_VISUAL:
        pygame.quit()
    
    print("\n✅  Complete!\n")
