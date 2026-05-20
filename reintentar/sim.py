"""
simulador_enhanced.py — Advanced Traffic Jam Simulator
=======================================================
Enhanced version with:
- 6-state system (ESTABILIZANDO, FLUJO_SINCRO, FRENADA_ORIGEN, ONDA_ACTIVA, DISOLVIENDO, RECUPERADO)
- Full parameter tunability with batch execution
- Speed variance (v_diff) tracking for better jam detection
- Comprehensive CSV logging with all relevant metrics
- Color-coded visualization for all states

Usage:
    python simulador_enhanced.py
"""

import pygame
import math
import csv
import time
import os

# ═══════════════════════════════════════════════════════════════════════════
#  BATCH EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
ESCENARIOS = [
    {
        "nombre": "Exp_Base_V25_D12",
        "num_cars": 27,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": True
    },
    {
        "nombre": "Exp_High_Density_V30_D10",
        "num_cars": 35,
        "car_max_speed": 30.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -10.0,
        "enable_visual": True
    },
    {
        "nombre": "Exp_Safe_V20_D8",
        "num_cars": 25,
        "car_max_speed": 20.0,
        "car_acc_max": 2.0,
        "car_dec_max": 4.0,
        "s0": 3.5,
        "t_reaction": 1.5,
        "disturbance_start": 10.0,
        "disturbance_duration": 4.0,
        "disturbance_decel": -8.0,
        "enable_visual": True
    }
]

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANT PARAMETERS (Usually don't need tuning)
# ═══════════════════════════════════════════════════════════════════════════
TRACK_RADIUS = 400                  # meters (keep constant for comparability)
CAR_LENGTH = 5.0                    # meters (vehicle physical length)
DT = 0.1                           # simulation timestep (seconds)
MAX_TIME = 300                     # maximum simulation time (seconds) - perpetual jam cutoff

# Calculate track length
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS


# ═══════════════════════════════════════════════════════════════════════════
#  ENHANCED SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════
class SimuladorEnhanced:
    """
    Advanced traffic simulator with 6-state system and comprehensive metrics.
    
    States:
    1. ESTABILIZANDO - Initial acceleration to equilibrium
    2. FLUJO_SINCRO - Stable synchronized flow (all cars at similar speed)
    3. FRENADA_ORIGEN - Disturbance vehicle braking
    4. ONDA_ACTIVA - Shockwave propagating (vehicles stopped)
    5. DISOLVIENDO - Queue dissolving (no stopped vehicles but accordion effect)
    6. RECUPERADO - System fully recovered to stable flow
    """
    
    def __init__(self, config):
        """Initialize simulator with given configuration."""
        self.config = config
        
        # Extract parameters
        self.num_cars = config['num_cars']
        self.car_max_speed = config['car_max_speed']
        self.car_acc_max = config['car_acc_max']
        self.car_dec_max = config['car_dec_max']
        self.s0 = config['s0']
        self.t_reaction = config['t_reaction']
        self.disturbance_start = config['disturbance_start']
        self.disturbance_duration = config['disturbance_duration']
        self.disturbance_decel = config['disturbance_decel']
        self.enable_visual = config['enable_visual']
        
        # Initialize pygame if visualization enabled
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"TFG Enhanced - {config['nombre']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 15)
        
        # Vehicle state arrays
        spacing = TRACK_LENGTH / self.num_cars
        self.positions = [i * spacing for i in range(self.num_cars)]
        self.velocities = [0.0] * self.num_cars
        self.accelerations = [0.0] * self.num_cars
        self.car_states = ["ESTABILIZANDO"] * self.num_cars  # Individual car state
        
        # Simulation state
        self.estado = "ESTABILIZANDO"
        self.segundo_actual = 0
        self.frame_count = 0
        self.timer_frenada = 0
        self.timer_fin = 0
        
        # Metrics tracking
        self.logs = []
        self.max_stopped = 0
        self.max_v_diff = 0
        self.t_dissolve = None
        self.jam_perpetuo = False
        
        # Print configuration
        print("\n" + "=" * 80)
        print(f"  ESCENARIO: {config['nombre']}")
        print("=" * 80)
        print(f"  Vehicles:          {self.num_cars}")
        print(f"  Track Length:      {TRACK_LENGTH:.0f} m")
        print(f"  Max Speed:         {self.car_max_speed} m/s ({self.car_max_speed * 3.6:.1f} km/h)")
        print(f"  Acceleration:      {self.car_acc_max} m/s²")
        print(f"  Braking:           {self.car_dec_max} m/s²")
        print(f"  Min Gap (S0):      {self.s0} m")
        print(f"  Reaction Time:     {self.t_reaction} s")
        print(f"  Disturbance:       {self.disturbance_decel} m/s² for {self.disturbance_duration}s at t={self.disturbance_start}s")
        print(f"  Visualization:     {self.enable_visual}")
        print("=" * 80 + "\n")

    def calculate_idm_acceleration(self, i):
        """Calculate IDM (Intelligent Driver Model) acceleration for vehicle i."""
        lead = (i + 1) % self.num_cars
        
        # Calculate gap to leader
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - CAR_LENGTH)
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        
        # Free flow term (desire to reach max speed)
        if self.car_max_speed > 0:
            v_ratio = v / self.car_max_speed
        else:
            v_ratio = 0
        free_term = self.car_acc_max * (1.0 - v_ratio ** 4)
        
        # Interaction term (maintaining safe distance)
        s_desired = self.s0 + max(0, v * self.t_reaction + 
                                  (v * dv) / (2 * math.sqrt(self.car_acc_max * self.car_dec_max)))
        interaction = self.car_acc_max * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        
        # Limit to vehicle capabilities
        accel = max(-self.car_dec_max, min(self.car_acc_max, accel))
        
        return accel

    def classify_car_state(self, i):
        """Classify individual car state based on velocity and acceleration."""
        v = self.velocities[i]
        a = self.accelerations[i]
        
        # Determine state based on current behavior
        if v < 0.5:
            return "PARADO"  # Stopped
        elif v < self.car_max_speed * 0.3:
            return "ATASCO"  # In jam (very slow)
        elif a < -2.0:
            return "FRENANDO"  # Hard braking
        elif v >= self.car_max_speed - 1.0:
            return "CRUCERO"  # Cruising at speed
        elif a > 1.0:
            return "ACELERANDO"  # Accelerating
        else:
            return "NORMAL"  # Normal driving

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        
        # Calculate accelerations based on current state
        if self.estado == "ESTABILIZANDO":
            # Gentle acceleration to cruising speed
            for i in range(self.num_cars):
                if self.velocities[i] < self.car_max_speed:
                    self.accelerations[i] = self.car_acc_max * 0.5
                else:
                    self.accelerations[i] = 0.0
        else:
            # IDM control for all vehicles
            for i in range(self.num_cars):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance to lead vehicle
        if self.estado == "FRENADA_ORIGEN":
            self.accelerations[0] = self.disturbance_decel
        
        # Update velocities and positions
        for i in range(self.num_cars):
            self.velocities[i] += self.accelerations[i] * DT
            
            # 🛑 REALISTIC STOP: Force dead stop when crawling and braking
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
                self.accelerations[i] = 0.0
            
            # Clamp velocity to valid range
            self.velocities[i] = max(0.0, min(self.car_max_speed, self.velocities[i]))
            
            # Update position
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH
            
            # Classify individual car state
            self.car_states[i] = self.classify_car_state(i)

    def log_data(self):
        """Record comprehensive statistics for this second."""
        
        # Basic velocity statistics
        avg_velocity = sum(self.velocities) / self.num_cars
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        v_diff = max_velocity - min_velocity  # Speed variance
        
        # Update max variance
        if v_diff > self.max_v_diff:
            self.max_v_diff = v_diff
        
        # Vehicle state counts
        state_counts = {
            "PARADO": sum(1 for s in self.car_states if s == "PARADO"),
            "ATASCO": sum(1 for s in self.car_states if s == "ATASCO"),
            "FRENANDO": sum(1 for s in self.car_states if s == "FRENANDO"),
            "NORMAL": sum(1 for s in self.car_states if s == "NORMAL"),
            "ACELERANDO": sum(1 for s in self.car_states if s == "ACELERANDO"),
            "CRUCERO": sum(1 for s in self.car_states if s == "CRUCERO")
        }
        
        stopped = state_counts["PARADO"]
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        # Gap statistics
        gaps = []
        for i in range(self.num_cars):
            lead = (i + 1) % self.num_cars
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
            gaps.append(gap)
        
        min_gap = min(gaps)
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        
        # Acceleration statistics
        positive_accels = [a for a in self.accelerations if a > 0]
        negative_accels = [a for a in self.accelerations if a < 0]
        avg_accel = sum(self.accelerations) / len(self.accelerations)
        
        # Log everything
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado_sistema": self.estado,
            
            # Velocity metrics
            "vel_media": round(avg_velocity, 2),
            "vel_min": round(min_velocity, 2),
            "vel_max": round(max_velocity, 2),
            "v_diff": round(v_diff, 2),
            
            # Vehicle state distribution
            "coches_parados": state_counts["PARADO"],
            "coches_atasco": state_counts["ATASCO"],
            "coches_frenando": state_counts["FRENANDO"],
            "coches_normal": state_counts["NORMAL"],
            "coches_acelerando": state_counts["ACELERANDO"],
            "coches_crucero": state_counts["CRUCERO"],
            
            # Gap metrics
            "gap_min": round(min_gap, 2),
            "gap_medio": round(avg_gap, 2),
            "gap_max": round(max_gap, 2),
            
            # Acceleration metrics
            "accel_media": round(avg_accel, 2),
            "coches_acelerando_count": len(positive_accels),
            "coches_frenando_count": len(negative_accels),
            
            # System metrics
            "max_stopped_acumulado": self.max_stopped,
            "max_v_diff_acumulado": round(self.max_v_diff, 2)
        })

    def get_state_color(self, state):
        """
        Get color for visualization based on 6-state system.
        
        Color scheme:
        - PARADO: Dark Red (stopped)
        - ATASCO: Red (in jam)
        - FRENANDO: Orange (braking)
        - NORMAL: Yellow (coasting)
        - ACELERANDO: Light Green (accelerating)
        - CRUCERO: Bright Green (cruising)
        """
        colors = {
            "PARADO": (139, 0, 0),        # Dark red
            "ATASCO": (255, 50, 50),      # Red
            "FRENANDO": (255, 140, 0),    # Orange
            "NORMAL": (255, 255, 0),      # Yellow
            "ACELERANDO": (144, 238, 144), # Light green
            "CRUCERO": (50, 255, 50),     # Bright green
            "ESTABILIZANDO": (100, 100, 100)  # Gray (initial)
        }
        return colors.get(state, (255, 255, 255))

    def draw(self):
        """Render simulation visualization."""
        if not self.enable_visual:
            return
        
        self.screen.fill((20, 20, 25))
        
        # Draw track
        center_x, center_y = 600, 475
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), 
                          TRACK_RADIUS, 30)
        
        # Draw vehicles with state-based colors
        for i in range(self.num_cars):
            angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
            x = center_x + TRACK_RADIUS * math.cos(angle)
            y = center_y + TRACK_RADIUS * math.sin(angle)
            
            color = self.get_state_color(self.car_states[i])
            
            # White outline for disturbance source (car 0)
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), 11)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
        
        # Statistics panel
        avg_vel = sum(self.velocities) / self.num_cars
        v_diff = max(self.velocities) - min(self.velocities)
        
        # Count states
        state_counts = {}
        for state in self.car_states:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        info = [
            f"=== {self.config['nombre']} ===",
            f"Time: {self.segundo_actual}s   System State: {self.estado}",
            "",
            "=== PARAMETERS ===",
            f"Vehicles:      {self.num_cars}",
            f"Max Speed:     {self.car_max_speed} m/s ({self.car_max_speed * 3.6:.0f} km/h)",
            f"Acceleration:  {self.car_acc_max} m/s²",
            f"Braking:       {self.car_dec_max} m/s²",
            f"Min Gap (S0):  {self.s0} m",
            f"Reaction:      {self.t_reaction} s",
            f"Disturbance:   {self.disturbance_decel} m/s² × {self.disturbance_duration}s",
            "",
            "=== CURRENT METRICS ===",
            f"Avg velocity:  {avg_vel:.2f} m/s",
            f"Min velocity:  {min(self.velocities):.2f} m/s",
            f"Max velocity:  {max(self.velocities):.2f} m/s",
            f"Speed Variance (v_diff): {v_diff:.2f} m/s",
            "",
            "=== VEHICLE DISTRIBUTION ===",
            f"🔴 Parados:     {state_counts.get('PARADO', 0)}",
            f"🔴 Atasco:      {state_counts.get('ATASCO', 0)}",
            f"🟠 Frenando:    {state_counts.get('FRENANDO', 0)}",
            f"🟡 Normal:      {state_counts.get('NORMAL', 0)}",
            f"🟢 Acelerando:  {state_counts.get('ACELERANDO', 0)}",
            f"🟢 Crucero:     {state_counts.get('CRUCERO', 0)}",
            "",
            f"Max stopped:    {self.max_stopped}",
            f"Max v_diff:     {self.max_v_diff:.2f}",
            "",
            "=== RESULTS ===",
        ]
        
        if self.t_dissolve:
            info.append(f"✅ Jam dissolved at {self.t_dissolve}s")
        elif self.jam_perpetuo:
            info.append("❌ PERPETUAL JAM!")
        else:
            info.append("⏳ Simulating...")
        
        for idx, text in enumerate(info):
            if text == "":
                continue
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 15 + idx * 18))
        
        # Legend
        legend_x, legend_y = 900, 750
        legend_items = [
            ((139, 0, 0), "Parado", 8),
            ((255, 50, 50), "Atasco", 8),
            ((255, 140, 0), "Frenando", 8),
            ((255, 255, 0), "Normal", 8),
            ((144, 238, 144), "Acelerando", 8),
            ((50, 255, 50), "Crucero", 8),
        ]
        
        for i, (color, label, size) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            pygame.draw.circle(self.screen, color, (legend_x, y_pos), size)
            self.screen.blit(self.font.render(label, True, (200, 200, 200)),
                            (legend_x + 20, y_pos - 8))
        
        pygame.display.flip()

    def save_csv(self):
        """Save comprehensive time-series data to CSV."""
        os.makedirs("resultados_enhanced", exist_ok=True)
        
        filename = f"resultados_enhanced/{self.config['nombre']}_data.csv"
        
        if not self.logs:
            print(f"  ⚠️  No data to save")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.logs[0].keys())
            writer.writeheader()
            writer.writerows(self.logs)
        
        print(f"  ✅  Data saved: {filename}")

    def run(self):
        """Main simulation loop with 6-state transitions."""
        running = True
        
        while running and self.segundo_actual < MAX_TIME:
            # Handle pygame events
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # Calculate key metrics for state transitions
            v_min = min(self.velocities)
            v_max = max(self.velocities)
            v_diff = v_max - v_min  # Speed variance
            equilibrium_speed = sum(self.velocities) / self.num_cars

            # ═══════════════════════════════════════════════════════════
            #  STATE TRANSITION LOGIC (6 States)
            # ═══════════════════════════════════════════════════════════
            
            # 1. ESTABILIZANDO -> FLUJO_SINCRO
            if self.estado == "ESTABILIZANDO":
                if v_diff < 1.0 and self.segundo_actual > 5:
                    self.estado = "FLUJO_SINCRO"
                    print(f"[{self.segundo_actual}s] 🟢 STABLE FLOW at {equilibrium_speed:.1f} m/s")

            # 2. FLUJO_SINCRO -> FRENADA_ORIGEN
            if self.estado == "FLUJO_SINCRO" and self.segundo_actual >= self.disturbance_start:
                self.estado = "FRENADA_ORIGEN"
                self.timer_frenada = self.disturbance_duration
                print(f"[{self.segundo_actual}s] 🚨 DISTURBANCE STARTED")

            # 3. FRENADA_ORIGEN -> ONDA_ACTIVA
            if self.estado == "FRENADA_ORIGEN":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "ONDA_ACTIVA"
                    print(f"[{self.segundo_actual}s] 💥 SHOCKWAVE PROPAGATING")

            # 4 & 5. ONDA_ACTIVA <-> DISOLVIENDO (bidirectional)
            if self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                # If someone fully stopped, wave is still active
                if v_min < 1.0 and self.estado == "DISOLVIENDO":
                    self.estado = "ONDA_ACTIVA"
                    print(f"[{self.segundo_actual}s] ⚠️  WAVE RE-FORMED")
                
                # If no full stops but still accordion effect
                elif v_min >= 1.0 and self.estado == "ONDA_ACTIVA":
                    self.estado = "DISOLVIENDO"
                    print(f"[{self.segundo_actual}s] 📉 QUEUE CLEARED - dissolving phase")

                # 6. DISOLVIENDO -> RECUPERADO
                # System recovered when speed variance is small again
                if self.estado == "DISOLVIENDO" and v_diff < 2.0:
                    if self.t_dissolve is None:
                        self.estado = "RECUPERADO"
                        self.t_dissolve = self.segundo_actual
                        final_speed = sum(self.velocities) / self.num_cars
                        print(f"[{self.segundo_actual}s] ✅ JAM DISSOLVED! Speed: {final_speed:.1f} m/s")
                        self.timer_fin = 5

            # Wait 5 seconds after recovery before ending
            if self.estado == "RECUPERADO":
                self.timer_fin -= DT
                if self.timer_fin <= 0:
                    running = False
            
            # Perpetual jam detection
            if self.segundo_actual > 200 and self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                stopped_cars = sum(1 for s in self.car_states if s == "PARADO")
                if stopped_cars > self.num_cars * 0.2:
                    print(f"[{self.segundo_actual}s] ❌ PERPETUAL JAM DETECTED!")
                    self.jam_perpetuo = True
                    running = False
            
            # Physics update
            self.step_physics()
            
            # Data logging (every second)
            self.frame_count += 1
            if self.frame_count >= 10:  # 10 frames * 0.1s = 1 second
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
            
            # Visualization
            if self.enable_visual:
                self.draw()
                self.clock.tick(60)
        
        # Save results
        self.save_csv()
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"  SIMULATION COMPLETE: {self.config['nombre']}")
        print("=" * 80)
        if self.t_dissolve:
            print(f"  ✅ SUCCESS: Jam dissolved in {self.t_dissolve}s")
            print(f"  Max stopped vehicles: {self.max_stopped}")
            print(f"  Max speed variance: {self.max_v_diff:.2f} m/s")
        elif self.jam_perpetuo:
            print(f"  ❌ FAILURE: Perpetual jam")
            print(f"  Max stopped: {self.max_stopped}")
            print(f"  Suggestion: Reduce density or soften disturbance")
        else:
            print(f"  ⏱️  Simulation ended at {self.segundo_actual}s")
        print("=" * 80 + "\n")
        
        return "NEXT"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION - Run all scenarios
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ENHANCED TRAFFIC JAM SIMULATOR" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"\n▶ Running scenario {i}/{len(ESCENARIOS)}: {config['nombre']}")
        sim = SimuladorEnhanced(config)
        result = sim.run()
        
        if result == "QUIT":
            print("\n⚠️  User interrupted simulation")
            break
    
    if ESCENARIOS[0]['enable_visual']:
        pygame.quit()
    
    print("\n" + "=" * 80)
    print("  ALL SIMULATIONS COMPLETE!")
    print("  📊 Results saved in: resultados_enhanced/")
    print("=" * 80)
    print("\n💡 TIP: Edit ESCENARIOS at the top to test different parameters!\n")
