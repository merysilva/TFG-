"""
simulador_DEBUG.py — Debug Version with Extensive Logging
=========================================================
This will print what's happening each second so we can see the issue.

Usage:
    python simulador_DEBUG.py
"""

import pygame
import math
import csv
import os

# Configuration
ENABLE_VISUAL = False
VISUAL_SPEED = 30

# Test just ONE scenario
ESCENARIOS = [
    {"nombre": "TEST_N25_V30", "num_cars": 25, "max_speed": 30.0}
]

# Physical constants
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


class SimuladorDebug:
    def __init__(self, config, enable_visual=True):
        self.enable_visual = enable_visual
        
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"DEBUG — {config['nombre']}")
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
        self.consecutive_clear_seconds = 0
        
        print(f"\n🚀 INITIALIZED")
        print(f"   Vehicles: {self.num_cars}")
        print(f"   Max speed: {self.max_speed} m/s")
        print(f"   Disturbance: {DISTURBANCE_START}s to {DISTURBANCE_START + DISTURBANCE_DURATION}s")
        print(f"   Max time: {MAX_TIME}s\n")

    def calculate_idm_acceleration(self, i):
        lead = (i + 1) % self.num_cars
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - CAR_LENGTH)
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        
        v_ratio = v / self.max_speed if self.max_speed > 0 else 0
        free_term = CAR_ACC_MAX * (1.0 - v_ratio ** 2)
        
        s_desired = S0 + max(0, v * T_REACTION + 
                            (v * dv) / (2 * math.sqrt(CAR_ACC_MAX * CAR_DEC_MAX)))
        interaction = CAR_ACC_MAX * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        accel = max(-CAR_DEC_MAX, min(CAR_ACC_MAX, accel))
        
        return accel

    def step_physics(self):
        for i in range(self.num_cars):
            self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        if self.disturbance_active:
            self.accelerations[0] = DISTURBANCE_DECEL
        
        for i in range(self.num_cars):
            self.velocities[i] += self.accelerations[i] * DT
            
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
            
            self.velocities[i] = max(0.0, min(self.max_speed, self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH

    def log_data(self):
        avg_velocity = sum(self.velocities) / self.num_cars
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        gaps = []
        for i in range(self.num_cars):
            lead = (i + 1) % self.num_cars
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
            gaps.append(gap)
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "disturbance_active": int(self.disturbance_active),
            "avg_velocity": round(avg_velocity, 2),
            "stopped": stopped,
            "max_stopped": self.max_stopped,
            "consecutive_clear": self.consecutive_clear_seconds,
        })

    def run(self):
        running = True
        iteration = 0
        
        print("⏱️  STARTING SIMULATION LOOP\n")
        
        while running and self.segundo_actual < MAX_TIME:
            iteration += 1
            
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("🛑 User quit")
                        return "QUIT"
            
            # Physics update
            self.step_physics()
            
            # Data logging every second
            self.frame_count += 1
            if self.frame_count >= 10:
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                stopped = sum(1 for v in self.velocities if v < 1.0)
                avg_v = sum(self.velocities) / self.num_cars
                
                # Print every second
                status = []
                
                # Disturbance control
                if self.segundo_actual == DISTURBANCE_START:
                    self.disturbance_active = True
                    self.disturbance_end_time = self.segundo_actual + DISTURBANCE_DURATION
                    status.append("🚨 DISTURBANCE START")
                
                if self.segundo_actual == self.disturbance_end_time:
                    self.disturbance_active = False
                    status.append("✋ DISTURBANCE END")
                
                # Check for dissolution
                if self.segundo_actual > self.disturbance_end_time:
                    if stopped == 0:
                        self.consecutive_clear_seconds += 1
                        status.append(f"✓ Clear {self.consecutive_clear_seconds}/5")
                        
                        if self.consecutive_clear_seconds >= 5 and self.t_dissolve is None:
                            self.t_dissolve = self.segundo_actual - 5
                            status.append(f"✅ DISSOLVED at {self.t_dissolve}s")
                            print(f"[{self.segundo_actual:3d}s] v={avg_v:5.2f} stopped={stopped:2d}/{self.max_stopped:2d}  {' '.join(status)}")
                            running = False
                            continue
                    else:
                        self.consecutive_clear_seconds = 0
                
                # Print status
                status_str = ' '.join(status) if status else ''
                print(f"[{self.segundo_actual:3d}s] v={avg_v:5.2f} stopped={stopped:2d}/{self.max_stopped:2d}  {status_str}")
            
            if self.enable_visual:
                pass  # Skip drawing for debug
        
        # Exit reasons
        print(f"\n📊 SIMULATION ENDED")
        print(f"   Final time: {self.segundo_actual}s")
        print(f"   Iterations: {iteration}")
        print(f"   Max stopped: {self.max_stopped}")
        print(f"   T_dissolve: {self.t_dissolve}")
        
        # Determine outcome
        if self.t_dissolve is None:
            if self.max_stopped > 0:
                self.jam_perpetuo = True
                print(f"   ❌ PERPETUAL JAM")
            else:
                print(f"   ⚠️  NO JAM FORMED")
        else:
            print(f"   ✅ DISSOLVED at {self.t_dissolve}s")
        
        # Save CSV
        os.makedirs("resultados_debug", exist_ok=True)
        filename = f"resultados_debug/{self.config['nombre']}_datos.csv"
        
        if self.logs:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.logs[0].keys())
                writer.writeheader()
                writer.writerows(self.logs)
            print(f"   💾 Saved: {filename}")
        
        summary = {
            "nombre": self.config["nombre"],
            "num_cars": self.config["num_cars"],
            "max_speed": self.config["max_speed"],
            "t_dissolve": self.t_dissolve if not self.jam_perpetuo else "NaN",
            "max_stopped": self.max_stopped,
            "jam_perpetuo": int(self.jam_perpetuo),
        }
        
        return "NEXT", summary


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DEBUG HOMOGENEOUS TRAFFIC SIMULATION")
    print("=" * 70)
    
    for config in ESCENARIOS:
        print(f"\n▶️  Running: {config['nombre']}")
        
        sim = SimuladorDebug(config, enable_visual=ENABLE_VISUAL)
        result, summary = sim.run()
        
        print(f"\n📋 SUMMARY:")
        print(f"   t_dissolve: {summary['t_dissolve']}")
        print(f"   max_stopped: {summary['max_stopped']}")
        print(f"   jam_perpetuo: {summary['jam_perpetuo']}")
        
        if result == "QUIT":
            break
    
    if ENABLE_VISUAL:
        pygame.quit()
    
    print("\n" + "=" * 70)
    print("  DEBUG COMPLETE")
    print("=" * 70 + "\n")