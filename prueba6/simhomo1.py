"""
simulador_homogeneo_FIXED.py — Actually Fixed Homogeneous Traffic
==================================================================
CRITICAL FIX: Don't check for dissolution before disturbance ends!

Usage:
    python simulador_homogeneo_FIXED.py
"""

import pygame
import math
import csv
import os

# Configuration
ENABLE_VISUAL = False
VISUAL_SPEED = 30

# Experiment configuration
DENSITIES = [20, 25, 27, 30, 35]
SPEEDS = [20.0, 25.0, 30.0, 35.0]

ESCENARIOS = []
for density in DENSITIES:
    for speed in SPEEDS:
        ESCENARIOS.append({
            "nombre": f"Homo_N{density:02d}_V{int(speed):02d}",
            "num_cars": density,
            "max_speed": speed
        })

# Physical constants (from working tuning)
TRACK_RADIUS = 400
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS
CAR_LENGTH = 5.0
CAR_ACC_MAX = 2.5
CAR_DEC_MAX = 4.5
S0 = 2.5
T_REACTION = 1.0

# Disturbance
DISTURBANCE_START = 10.0
DISTURBANCE_DURATION = 5.0
DISTURBANCE_DECEL = -12.0

# Simulation
MAX_TIME = 300
DT = 0.1


class SimuladorHomogeneo:
    def __init__(self, config, enable_visual=True):
        self.enable_visual = enable_visual
        
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"TFG — {config['nombre']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 15)
        
        self.config = config
        self.num_cars = config["num_cars"]
        self.max_speed = config["max_speed"]
        
        # Initial state
        spacing = TRACK_LENGTH / self.num_cars
        self.positions = [i * spacing for i in range(self.num_cars)]
        self.velocities = [0.0] * self.num_cars
        self.accelerations = [0.0] * self.num_cars
        
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
        
        # Tracking for dissolution detection
        self.stable_counter = 0
        self.jam_occurred = False  # NEW: Track if a jam actually happened

    def calculate_idm_acceleration(self, i):
        lead = (i + 1) % self.num_cars
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - CAR_LENGTH)
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        
        # Free flow term (v^2 from working config)
        v_ratio = v / self.max_speed if self.max_speed > 0 else 0
        free_term = CAR_ACC_MAX * (1.0 - v_ratio ** 2)
        
        # Interaction term
        s_desired = S0 + max(0, v * T_REACTION + 
                            (v * dv) / (2 * math.sqrt(CAR_ACC_MAX * CAR_DEC_MAX)))
        interaction = CAR_ACC_MAX * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        accel = max(-CAR_DEC_MAX, min(CAR_ACC_MAX, accel))
        
        return accel

    def step_physics(self):
        # Calculate accelerations
        for i in range(self.num_cars):
            self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance
        if self.disturbance_active:
            self.accelerations[0] = DISTURBANCE_DECEL
        
        # Update velocities and positions
        for i in range(self.num_cars):
            self.velocities[i] += self.accelerations[i] * DT
            
            # Realistic stop
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
            
            self.velocities[i] = max(0.0, min(self.max_speed, self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def log_data(self):
        avg_velocity = sum(self.velocities) / self.num_cars
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        v_diff = max_velocity - min_velocity
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        cruising = sum(1 for v in self.velocities if v >= self.max_speed - 1.0)
        braking = sum(1 for a in self.accelerations if a < -1.0)
        accelerating = sum(1 for a in self.accelerations if a > 0.5)
        
        # Track maximum stopped
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        # Track if a jam occurred
        if stopped > 0 and self.segundo_actual > DISTURBANCE_START:
            self.jam_occurred = True
        
        # Calculate gaps
        gaps = []
        for i in range(self.num_cars):
            lead = (i + 1) % self.num_cars
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
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
            "num_cars": self.num_cars,
            "max_speed": self.max_speed,
            "avg_velocity": round(avg_velocity, 2),
            "min_velocity": round(min_velocity, 2),
            "max_velocity": round(max_velocity, 2),
            "v_diff": round(v_diff, 2),
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
        
        for i in range(self.num_cars):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_vehicle_color(i)
            
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), 10)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        avg_vel = sum(self.velocities) / self.num_cars
        
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
        os.makedirs("resultados_homogeneo", exist_ok=True)
        filename = f"resultados_homogeneo/{self.config['nombre']}_datos.csv"
        
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
                    if stopped > self.num_cars * 0.10:
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
            "num_cars": self.config["num_cars"],
            "max_speed": self.config["max_speed"],
            "density": round(self.num_cars / TRACK_LENGTH * 1000, 2),
            "t_dissolve": self.t_dissolve if not self.jam_perpetuo else "NaN",
            "max_stopped": self.max_stopped,
            "jam_perpetuo": int(self.jam_perpetuo),
        }
        
        return "NEXT", summary


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HOMOGENEOUS TRAFFIC SIMULATION (ACTUALLY FIXED)")
    print("=" * 70)
    print(f"  Scenarios: {len(ESCENARIOS)}")
    print(f"  Visualization: {'ENABLED' if ENABLE_VISUAL else 'DISABLED'}")
    print("=" * 70 + "\n")
    
    master_results = []
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i:02d}/{len(ESCENARIOS)}] {config['nombre']} ", end="", flush=True)
        
        sim = SimuladorHomogeneo(config, enable_visual=ENABLE_VISUAL)
        result, summary = sim.run()
        
        master_results.append(summary)
        
        if result == "QUIT":
            print("\n⛔  Interrupted")
            break
        
        print(f"✓ T_dissolve={summary['t_dissolve']}s  Max_stopped={summary['max_stopped']}")
    
    if master_results:
        with open("resultados_homogeneo_master.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_results[0].keys())
            writer.writeheader()
            writer.writerows(master_results)
        print(f"\n✅  Master: resultados_homogeneo_master.csv")
    
    if ENABLE_VISUAL:
        pygame.quit()
    
    print("\n✅  Complete!\n")
