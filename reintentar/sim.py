"""
simulador_enhanced.py — Advanced Traffic Jam Simulator (v2)
============================================================
Enhanced version with:
- Start at cruise speed (realistic initial conditions)
- 4-state system: PARADO, ACELERANDO, DECELERANDO, CONSTANTE
- Advanced metrics: flow, density, wave propagation, efficiency
- Full parameter tunability with batch execution
- Clean console output

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
    # Baseline scenarios
    {
        "nombre": "Base_N25_V25_D10",
        "num_cars": 25,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -10.0,
        "enable_visual": False
    },
    {
        "nombre": "Base_N30_V25_D10",
        "num_cars": 30,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -10.0,
        "enable_visual": False
    },
    {
        "nombre": "Base_N35_V25_D10",
        "num_cars": 35,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -10.0,
        "enable_visual": False
    },
    # High speed scenarios
    {
        "nombre": "HighSpeed_N25_V30_D12",
        "num_cars": 25,
        "car_max_speed": 30.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "HighSpeed_N30_V30_D12",
        "num_cars": 30,
        "car_max_speed": 30.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    # Conservative (safe) scenarios
    {
        "nombre": "Safe_N25_V20_S4_T15",
        "num_cars": 25,
        "car_max_speed": 20.0,
        "car_acc_max": 2.0,
        "car_dec_max": 4.0,
        "s0": 4.0,
        "t_reaction": 1.5,
        "disturbance_start": 10.0,
        "disturbance_duration": 4.0,
        "disturbance_decel": -8.0,
        "enable_visual": False
    },
    {
        "nombre": "Safe_N30_V20_S4_T15",
        "num_cars": 30,
        "car_max_speed": 20.0,
        "car_acc_max": 2.0,
        "car_dec_max": 4.0,
        "s0": 4.0,
        "t_reaction": 1.5,
        "disturbance_start": 10.0,
        "disturbance_duration": 4.0,
        "disturbance_decel": -8.0,
        "enable_visual": False
    },
    # Extreme scenarios
    {
        "nombre": "Extreme_Dense_N40_V25_D15",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 6.0,
        "disturbance_decel": -15.0,
        "enable_visual": False
    },
    {
        "nombre": "Extreme_HardBrake_N30_V25_D18",
        "num_cars": 30,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 3.0,
        "disturbance_decel": -18.0,
        "enable_visual": False
    },
    # Visual demo (enable this one to watch)
    {
        "nombre": "VISUAL_Demo_N27_V25",
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
    }
]


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
TRACK_RADIUS = 400                  # meters
CAR_LENGTH = 5.0                    # meters
DT = 0.1                           # simulation timestep (seconds)
MAX_TIME = 300                     # maximum simulation time (seconds)

# Calculate track length
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS

# State classification thresholds
ACCEL_THRESHOLD = 0.5              # m/s² - above this = accelerating
DECEL_THRESHOLD = -0.5             # m/s² - below this = decelerating
STOPPED_THRESHOLD = 0.5            # m/s - below this = stopped


# ═══════════════════════════════════════════════════════════════════════════
#  ENHANCED SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════
class SimuladorEnhanced:
    """
    Advanced traffic simulator with realistic initial conditions and comprehensive metrics.
    
    4 States (based on acceleration):
    - PARADO: v < 0.5 m/s
    - ACELERANDO: a > 0.5 m/s²
    - DECELERANDO: a < -0.5 m/s²
    - CONSTANTE: -0.5 <= a <= 0.5 m/s²
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
        
        # Calculate equilibrium speed and spacing
        # IDM equilibrium: vehicles maintain steady speed with safe gaps
        self.equilibrium_speed = self.car_max_speed * 0.85  # Start at ~85% of max
        self.equilibrium_gap = TRACK_LENGTH / self.num_cars
        
        # Initialize pygame if visualization enabled
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"Traffic Simulator - {config['nombre']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 14)
        
        # Vehicle state arrays - START AT EQUILIBRIUM SPEED
        spacing = TRACK_LENGTH / self.num_cars
        self.positions = [i * spacing for i in range(self.num_cars)]
        self.velocities = [self.equilibrium_speed] * self.num_cars  # All start at equilibrium
        self.accelerations = [0.0] * self.num_cars
        self.car_states = ["CONSTANTE"] * self.num_cars
        
        # Simulation state
        self.estado = "EQUILIBRIO"  # Start in equilibrium, not stabilizing
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
        
        # Wave propagation tracking
        self.disturbance_initiated = False
        self.wave_front_car = None
        self.wave_propagation_speed = 0
        
        # Energy and efficiency tracking
        self.total_energy_dissipated = 0
        self.cumulative_time_loss = 0
        
        print(f"▶ {config['nombre']}: N={self.num_cars}, V={self.car_max_speed}m/s, S0={self.s0}m, Dist={self.disturbance_decel}m/s²")

    def calculate_idm_acceleration(self, i):
        """Calculate IDM acceleration for vehicle i."""
        lead = (i + 1) % self.num_cars
        
        # Calculate gap to leader
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - CAR_LENGTH)
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        
        # Free flow term
        v_ratio = v / self.car_max_speed if self.car_max_speed > 0 else 0
        free_term = self.car_acc_max * (1.0 - v_ratio ** 4)
        
        # Interaction term
        s_desired = self.s0 + max(0, v * self.t_reaction + 
                                  (v * dv) / (2 * math.sqrt(self.car_acc_max * self.car_dec_max)))
        interaction = self.car_acc_max * (s_desired / gap) ** 2
        
        accel = free_term - interaction
        
        # Limit to vehicle capabilities
        accel = max(-self.car_dec_max, min(self.car_acc_max, accel))
        
        return accel

    def classify_car_state(self, i):
        """Classify car state based on velocity and acceleration (4 states)."""
        v = self.velocities[i]
        a = self.accelerations[i]
        
        if v < STOPPED_THRESHOLD:
            return "PARADO"
        elif a > ACCEL_THRESHOLD:
            return "ACELERANDO"
        elif a < DECEL_THRESHOLD:
            return "DECELERANDO"
        else:
            return "CONSTANTE"

    def calculate_advanced_metrics(self):
        """Calculate flow, density, efficiency, and wave metrics."""
        # Basic stats
        avg_velocity = sum(self.velocities) / self.num_cars
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        v_diff = max_velocity - min_velocity
        
        # Density (vehicles per km)
        density = (self.num_cars / TRACK_LENGTH) * 1000  # veh/km
        
        # Flow (vehicles per hour passing a point)
        flow = density * avg_velocity * 3600 / 1000  # veh/hour
        
        # Efficiency (actual speed / target speed)
        efficiency = (avg_velocity / self.equilibrium_speed) * 100 if self.equilibrium_speed > 0 else 0
        
        # Time loss (seconds lost per vehicle compared to equilibrium)
        if self.equilibrium_speed > 0:
            ideal_distance = self.equilibrium_speed * DT
            actual_distance = avg_velocity * DT
            time_loss_per_car = (ideal_distance - actual_distance) / self.equilibrium_speed if actual_distance < ideal_distance else 0
            self.cumulative_time_loss += time_loss_per_car * self.num_cars
        
        # Energy dissipation (sum of braking energy)
        energy_this_step = sum(abs(min(0, a)) * v * DT for a, v in zip(self.accelerations, self.velocities))
        self.total_energy_dissipated += energy_this_step
        
        # Gap statistics
        gaps = []
        for i in range(self.num_cars):
            lead = (i + 1) % self.num_cars
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
            gaps.append(gap)
        
        # Wave propagation speed (if disturbance active)
        wave_speed_kmh = 0
        if self.estado in ["FRENADA_ORIGEN", "ONDA_ACTIVA", "DISOLVIENDO"]:
            # Find the rearmost stopped/slow vehicle
            slow_cars = [i for i, v in enumerate(self.velocities) if v < self.equilibrium_speed * 0.5]
            if slow_cars and self.wave_front_car is not None:
                # Wave travels backward (opposite to traffic direction)
                wave_distance = (self.positions[self.wave_front_car] - self.positions[slow_cars[-1]]) % TRACK_LENGTH
                time_elapsed = self.segundo_actual - self.disturbance_start
                if time_elapsed > 0:
                    wave_speed_kmh = -(wave_distance / time_elapsed) * 3.6  # negative = backward, km/h
        
        return {
            'avg_velocity': avg_velocity,
            'min_velocity': min_velocity,
            'max_velocity': max_velocity,
            'v_diff': v_diff,
            'density': density,
            'flow': flow,
            'efficiency': efficiency,
            'gaps': gaps,
            'energy_dissipated': self.total_energy_dissipated,
            'time_loss': self.cumulative_time_loss,
            'wave_speed_kmh': wave_speed_kmh
        }

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        
        # Calculate accelerations using IDM
        for i in range(self.num_cars):
            self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance to lead vehicle
        if self.estado == "FRENADA_ORIGEN":
            self.accelerations[0] = self.disturbance_decel
            if not self.disturbance_initiated:
                self.disturbance_initiated = True
                self.wave_front_car = 0
        
        # Update velocities and positions
        for i in range(self.num_cars):
            self.velocities[i] += self.accelerations[i] * DT
            
            # Realistic stop: force dead stop when crawling and braking
            if self.velocities[i] < STOPPED_THRESHOLD and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
            
            # Clamp velocity
            self.velocities[i] = max(0.0, min(self.car_max_speed, self.velocities[i]))
            
            # Update position
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH
            
            # Classify state
            self.car_states[i] = self.classify_car_state(i)

    def log_data(self):
        """Record comprehensive statistics."""
        metrics = self.calculate_advanced_metrics()
        
        # State counts
        state_counts = {
            "PARADO": self.car_states.count("PARADO"),
            "ACELERANDO": self.car_states.count("ACELERANDO"),
            "DECELERANDO": self.car_states.count("DECELERANDO"),
            "CONSTANTE": self.car_states.count("CONSTANTE")
        }
        
        stopped = state_counts["PARADO"]
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        if metrics['v_diff'] > self.max_v_diff:
            self.max_v_diff = metrics['v_diff']
        
        # Log everything
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado_sistema": self.estado,
            
            # Velocity metrics
            "vel_media": round(metrics['avg_velocity'], 2),
            "vel_min": round(metrics['min_velocity'], 2),
            "vel_max": round(metrics['max_velocity'], 2),
            "v_diff": round(metrics['v_diff'], 2),
            
            # Flow and efficiency
            "densidad_veh_km": round(metrics['density'], 2),
            "flujo_veh_h": round(metrics['flow'], 0),
            "eficiencia_pct": round(metrics['efficiency'], 1),
            
            # Vehicle states (4 states)
            "coches_parados": state_counts["PARADO"],
            "coches_acelerando": state_counts["ACELERANDO"],
            "coches_decelerando": state_counts["DECELERANDO"],
            "coches_constante": state_counts["CONSTANTE"],
            
            # Gap metrics
            "gap_min": round(min(metrics['gaps']), 2),
            "gap_medio": round(sum(metrics['gaps']) / len(metrics['gaps']), 2),
            "gap_max": round(max(metrics['gaps']), 2),
            
            # Advanced metrics
            "energia_disipada_acum": round(metrics['energy_dissipated'], 1),
            "tiempo_perdido_acum": round(metrics['time_loss'], 1),
            "vel_onda_kmh": round(metrics['wave_speed_kmh'], 1),
            
            # Cumulative extremes
            "max_parados": self.max_stopped,
            "max_v_diff": round(self.max_v_diff, 2)
        })

    def get_state_color(self, state):
        """Get color for 4-state visualization."""
        colors = {
            "PARADO": (200, 0, 0),        # Red
            "DECELERANDO": (255, 165, 0),  # Orange
            "CONSTANTE": (100, 200, 100),  # Green
            "ACELERANDO": (50, 150, 255),  # Blue
        }
        return colors.get(state, (255, 255, 255))

    def draw(self):
        """Render visualization."""
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
            
            color = self.get_state_color(self.car_states[i])
            
            # White outline for disturbance source
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), 11)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
        
        # Statistics panel
        metrics = self.calculate_advanced_metrics()
        state_counts = {s: self.car_states.count(s) for s in ["PARADO", "ACELERANDO", "DECELERANDO", "CONSTANTE"]}
        
        info = [
            f"=== {self.config['nombre']} ===",
            f"Time: {self.segundo_actual}s | State: {self.estado}",
            "",
            f"Velocity: {metrics['avg_velocity']:.1f} m/s (min: {metrics['min_velocity']:.1f}, max: {metrics['max_velocity']:.1f})",
            f"V-diff: {metrics['v_diff']:.1f} m/s | Efficiency: {metrics['efficiency']:.0f}%",
            f"Density: {metrics['density']:.1f} veh/km | Flow: {metrics['flow']:.0f} veh/h",
            "",
            f"🔴 Parados: {state_counts['PARADO']} | 🟠 Frenando: {state_counts['DECELERANDO']}",
            f"🟢 Constante: {state_counts['CONSTANTE']} | 🔵 Acelerando: {state_counts['ACELERANDO']}",
            "",
            f"Energy lost: {metrics['energy_dissipated']:.0f} | Time lost: {metrics['time_loss']:.0f}s",
            f"Wave speed: {metrics['wave_speed_kmh']:.1f} km/h",
        ]
        
        if self.t_dissolve:
            info.append(f"✅ Recovered at t={self.t_dissolve}s")
        elif self.jam_perpetuo:
            info.append("❌ PERPETUAL JAM")
        
        for idx, text in enumerate(info):
            if text == "":
                continue
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (20, 15 + idx * 20))
        
        # Legend
        legend_items = [
            ((200, 0, 0), "Parado"),
            ((255, 165, 0), "Decelerando"),
            ((100, 200, 100), "Constante"),
            ((50, 150, 255), "Acelerando"),
        ]
        
        for i, (color, label) in enumerate(legend_items):
            y_pos = 750 + i * 30
            pygame.draw.circle(self.screen, color, (900, y_pos), 8)
            self.screen.blit(self.font.render(label, True, (200, 200, 200)),
                            (920, y_pos - 8))
        
        pygame.display.flip()

    def save_csv(self):
        """Save CSV data."""
        os.makedirs("resultados_enhanced", exist_ok=True)
        filename = f"resultados_enhanced/{self.config['nombre']}_data.csv"
        
        if not self.logs:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.logs[0].keys())
            writer.writeheader()
            writer.writerows(self.logs)

    def run(self):
        """Main simulation loop."""
        running = True
        
        while running and self.segundo_actual < MAX_TIME:
            # Handle pygame events
            if self.enable_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # Calculate metrics
            v_min = min(self.velocities)
            v_diff = max(self.velocities) - v_min
            
            # State transitions (simplified)
            if self.estado == "EQUILIBRIO" and self.segundo_actual >= self.disturbance_start:
                self.estado = "FRENADA_ORIGEN"
                self.timer_frenada = self.disturbance_duration

            if self.estado == "FRENADA_ORIGEN":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "ONDA_ACTIVA"

            if self.estado == "ONDA_ACTIVA":
                if v_min >= 1.0:
                    self.estado = "DISOLVIENDO"

            if self.estado == "DISOLVIENDO":
                if v_min < 1.0:
                    self.estado = "ONDA_ACTIVA"
                elif v_diff < 2.0:
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
                stopped = sum(1 for s in self.car_states if s == "PARADO")
                if stopped > self.num_cars * 0.2:
                    self.jam_perpetuo = True
                    running = False
            
            # Physics update
            self.step_physics()
            
            # Logging
            self.frame_count += 1
            if self.frame_count >= 10:
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
            
            # Visualization
            if self.enable_visual:
                self.draw()
                self.clock.tick(60)
        
        # Save and report
        self.save_csv()
        
        if self.t_dissolve:
            print(f"   ✅ Dissolved at t={self.t_dissolve}s | Max stopped: {self.max_stopped}")
        elif self.jam_perpetuo:
            print(f"   ❌ Perpetual jam | Max stopped: {self.max_stopped}")
        else:
            print(f"   ⏱️  Timeout at {self.segundo_actual}s")
        
        return "NEXT"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 22 + "TRAFFIC JAM SIMULATOR v2" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\n📊 Running {len(ESCENARIOS)} scenarios...\n")
    
    start_time = time.time()
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i}/{len(ESCENARIOS)}] ", end="")
        sim = SimuladorEnhanced(config)
        result = sim.run()
        
        if result == "QUIT":
            print("\n⚠️  User interrupted simulation")
            break
    
    elapsed = time.time() - start_time
    
    print("\n" + "═" * 80)
    print(f"  ✅ All simulations complete! ({elapsed:.1f}s)")
    print(f"  📁 Results: resultados_enhanced/")
    print("═" * 80 + "\n")
