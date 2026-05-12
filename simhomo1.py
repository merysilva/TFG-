"""
simulador_homogeneo_densidad_velocidad.py — Homogeneous Traffic Experiments
============================================================================
Tests different vehicle densities and cruising speeds with only cars
(non-aggressive drivers). Uses the working parameters from the tuning tool.

Experiments:
    - Densities: 20, 25, 27, 30, 35 vehicles
    - Speeds: 20, 25, 30, 35 m/s
    - Total: 5 × 4 = 20 scenarios

Usage:
    python simulador_homogeneo_densidad_velocidad.py
"""

import pygame
import math
import csv
import os

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
ENABLE_VISUAL = False  # Set to False for faster execution
VISUAL_SPEED = 30      # FPS when visual is enabled

# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
DENSITIES = [20, 25, 27, 30, 35]  # Number of vehicles
SPEEDS = [20.0, 25.0, 30.0, 35.0]  # Maximum speeds (m/s)

# Generate all scenario combinations
ESCENARIOS = []
for density in DENSITIES:
    for speed in SPEEDS:
        ESCENARIOS.append({
            "nombre": f"Homo_N{density:02d}_V{int(speed):02d}",
            "num_cars": density,
            "max_speed": speed
        })

# ═══════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS (From working tuning parameters)
# ═══════════════════════════════════════════════════════════════════
TRACK_RADIUS = 400
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS

# Vehicle properties (cautious drivers, cars only)
CAR_LENGTH = 5.0
CAR_ACC_MAX = 2.5      # From working config
CAR_DEC_MAX = 4.5      # From working config

# IDM parameters (from working config)
S0 = 2.5               # minimum gap
T_REACTION = 1.0       # reaction time

# Disturbance parameters (from working config)
DISTURBANCE_START = 10.0
DISTURBANCE_DURATION = 5.0
DISTURBANCE_DECEL = -12.0

# Simulation parameters
MAX_TIME = 300
DT = 0.1


# ═══════════════════════════════════════════════════════════════════
#  SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════
class SimuladorHomogeneo:
    def __init__(self, config, enable_visual=True):
        """Initialize simulation with configuration."""
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
        
        # Initial positions (evenly spaced)
        spacing = TRACK_LENGTH / self.num_cars
        self.positions = [i * spacing for i in range(self.num_cars)]
        self.velocities = [0.0] * self.num_cars
        self.accelerations = [0.0] * self.num_cars
        
        # Simulation state
        self.estado = "ESTABILIZANDO"
        self.segundo_actual = 0
        self.frame_count = 0
        self.timer_frenada = 0
        self.timer_fin = 0
        
        # Metrics
        self.logs = []
        self.max_stopped = 0
        self.t_dissolve = None
        self.jam_perpetuo = False

    def calculate_idm_acceleration(self, i):
        """Calculate IDM acceleration for vehicle i."""
        lead = (i + 1) % self.num_cars
        
        # Calculate gap
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - CAR_LENGTH)
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        
        # Free flow term (using v^2 instead of v^4 as in working version)
        if self.max_speed > 0:
            v_ratio = v / self.max_speed
        else:
            v_ratio = 0
        free_term = CAR_ACC_MAX * (1.0 - v_ratio ** 2)
        
        # Interaction term
        s_desired = S0 + max(0, v * T_REACTION + 
                            (v * dv) / (2 * math.sqrt(CAR_ACC_MAX * CAR_DEC_MAX)))
        interaction = CAR_ACC_MAX * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        
        # Limit to vehicle capabilities
        accel = max(-CAR_DEC_MAX, min(CAR_ACC_MAX, accel))
        
        return accel

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        if self.estado == "ESTABILIZANDO":
            # Gentle acceleration to cruising speed
            for i in range(self.num_cars):
                if self.velocities[i] < self.max_speed:
                    self.accelerations[i] = CAR_ACC_MAX * 0.5
                else:
                    self.accelerations[i] = 0.0
        else:
            # IDM control
            for i in range(self.num_cars):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance
        if self.estado == "FRENADA_ORIGEN":
            self.accelerations[0] = DISTURBANCE_DECEL
        
        # Update velocities and positions
        for i in range(self.num_cars):
            self.velocities[i] += self.accelerations[i] * DT
            
            # Realistic stop: force dead stop when very slow and braking
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
                self.accelerations[i] = 0.0
                
            self.velocities[i] = max(0.0, min(self.max_speed, self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def log_data(self):
        """Record statistics for this second."""
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
        
        # Calculate gaps
        gaps = []
        for i in range(self.num_cars):
            lead = (i + 1) % self.num_cars
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
            gaps.append(gap)
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado": self.estado,
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
        """Calculate vehicle color based on acceleration."""
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
        """Render simulation visualization."""
        if not self.enable_visual:
            return
        
        self.screen.fill((20, 20, 25))
        
        # Draw track
        center_x, center_y = 600, 475
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), 
                          TRACK_RADIUS, 30)
        
        # Draw vehicles
        for i in range(self.num_cars):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_vehicle_color(i)
            
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), 10)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
        
        # Statistics panel
        avg_vel = sum(self.velocities) / self.num_cars
        stopped = sum(1 for v in self.velocities if v < 1.0)
        v_diff = max(self.velocities) - min(self.velocities)
        
        info = [
            f"SCENARIO: {self.config['nombre']}",
            f"Vehicles: {self.num_cars}  Max Speed: {self.max_speed} m/s",
            "",
            f"TIME: {self.segundo_actual}s   STATE: {self.estado}",
            "",
            "=== FLOW ===",
            f"Avg velocity:  {avg_vel:.2f} m/s",
            f"V_diff:        {v_diff:.2f} m/s",
            f"Stopped:       {stopped}",
            f"Max stopped:   {self.max_stopped}",
        ]
        
        if self.t_dissolve:
            info.append(f"✅ Dissolved: {self.t_dissolve}s")
        elif self.jam_perpetuo:
            info.append("❌ PERPETUAL JAM")
        
        for idx, text in enumerate(info):
            if text == "":
                continue
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 15 + idx * 18))
        
        pygame.display.flip()

    def save_csv(self):
        """Save time-series data to CSV."""
        os.makedirs("resultados_homogeneo", exist_ok=True)
        filename = f"resultados_homogeneo/{self.config['nombre']}_datos.csv"
        
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
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # Calculate metrics
            v_min = min(self.velocities)
            v_max = max(self.velocities)
            v_diff = v_max - v_min
            
            # State transitions (from working version)
            if self.estado == "ESTABILIZANDO":
                if v_diff < 1.0 and self.segundo_actual > 5:
                    self.estado = "FLUJO_SINCRO"
            
            if self.estado == "FLUJO_SINCRO" and self.segundo_actual >= DISTURBANCE_START:
                self.estado = "FRENADA_ORIGEN"
                self.timer_frenada = DISTURBANCE_DURATION
            
            if self.estado == "FRENADA_ORIGEN":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "ONDA_ACTIVA"
            
            if self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                if v_min < 1.0 and self.estado == "DISOLVIENDO":
                    self.estado = "ONDA_ACTIVA"
                elif v_min >= 1.0 and self.estado == "ONDA_ACTIVA":
                    self.estado = "DISOLVIENDO"
                
                if self.estado == "DISOLVIENDO" and v_diff < 2.0:
                    if self.t_dissolve is None:
                        self.estado = "RECUPERADO"
                        self.t_dissolve = self.segundo_actual
                        self.timer_fin = 5
            
            if self.estado == "RECUPERADO":
                self.timer_fin -= DT
                if self.timer_fin <= 0:
                    running = False
            
            # Perpetual jam detection
            if self.segundo_actual > 200 and self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                stopped_cars = sum(1 for v in self.velocities if v < 1.0)
                if stopped_cars > self.num_cars * 0.2:
                    self.jam_perpetuo = True
                    running = False
            
            # Physics update
            self.step_physics()
            
            # Data logging
            self.frame_count += 1
            if self.frame_count >= 10:
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
            
            # Visualization
            if self.enable_visual:
                self.draw()
                self.clock.tick(VISUAL_SPEED)
        
        self.save_csv()
        
        summary = {
            "nombre": self.config["nombre"],
            "num_cars": self.config["num_cars"],
            "max_speed": self.config["max_speed"],
            "density": round(self.num_cars / TRACK_LENGTH * 1000, 2),  # vehicles/km
            "t_dissolve": self.t_dissolve if not self.jam_perpetuo else "NaN",
            "max_stopped": self.max_stopped,
            "jam_perpetuo": int(self.jam_perpetuo),
        }
        
        return "NEXT", summary


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HOMOGENEOUS TRAFFIC SIMULATION (Density × Speed)")
    print("=" * 70)
    print(f"  Scenarios: {len(ESCENARIOS)}")
    print(f"  Densities: {DENSITIES}")
    print(f"  Speeds: {SPEEDS}")
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
        
        print(f"✓ T_dissolve={summary['t_dissolve']}s")
    
    # Save master results
    if master_results:
        with open("resultados_homogeneo_master.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_results[0].keys())
            writer.writeheader()
            writer.writerows(master_results)
        print(f"\n✅  Master: resultados_homogeneo_master.csv")
    
    if ENABLE_VISUAL:
        pygame.quit()
    
    print("\n✅  Simulation complete!\n")
