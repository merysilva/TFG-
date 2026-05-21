"""
sim.py — Traffic Jam Simulator
================================
Realistic traffic simulation with gap-based state classification.

Vehicle states based on spacing (not acceleration):
- ATASCO: Following too close (gap < critical distance)
- LIBRE: Normal following distance
- PARADO: Not moving (v < threshold)
- (Acceleration is implicit in the state transitions)

Usage:
    python sim.py
"""

import pygame
import math
import csv
import time
import os

# ═══════════════════════════════════════════════════════════════════════════
#  BATCH EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
"""
sim_focused.py — Focused Traffic Jam Experiments
=================================================
Simplified experimental design for thesis with controlled variables.

Three studies:
1. Density sweep (primary factor)
2. Behavior comparison (answers optimization question)
3. Disturbance sensitivity (secondary validation)
"""

# Copy all the imports and setup from sim.py here...
# (I'm just showing the ESCENARIOS part for brevity)

# ═══════════════════════════════════════════════════════════════════════════
#  FOCUSED EXPERIMENTAL DESIGN
# ═══════════════════════════════════════════════════════════════════════════

ESCENARIOS = [
    
    # ========================================================================
    # STUDY 1: DENSITY SWEEP (6 scenarios)
    # Variables: num_cars only
    # Fixed: V=25, s0=2.5, T=1.0, disturbance=-12 m/s²
    # ========================================================================
    {
        "nombre": "Density_N25",
        "num_cars": 25,
        "car_max_speed": 25.0,
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
        "nombre": "Density_N30",
        "num_cars": 30,
        "car_max_speed": 25.0,
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
        "nombre": "Density_N35",
        "num_cars": 35,
        "car_max_speed": 25.0,
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
        "nombre": "Density_N40",
        "num_cars": 40,
        "car_max_speed": 25.0,
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
        "nombre": "Density_N45",
        "num_cars": 45,
        "car_max_speed": 25.0,
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
        "nombre": "Density_N50",
        "num_cars": 50,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    
    # ========================================================================
    # STUDY 2: BEHAVIOR OPTIMIZATION (3 scenarios)
    # Variables: s0 and t_reaction combinations
    # Fixed: N=40, V=25, disturbance=-12 m/s²
    # Answers: "Which behavior minimizes stop-and-go?"
    # ========================================================================
    {
        "nombre": "Behavior_Aggressive",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.0,           # Tight following
        "t_reaction": 0.8,   # Quick reactions
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "Behavior_Normal",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,           # Standard gap
        "t_reaction": 1.0,   # Standard reaction
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    {
        "nombre": "Behavior_Conservative",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 4.0,           # Large safe gaps
        "t_reaction": 1.5,   # Cautious reactions
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,
        "enable_visual": False
    },
    
    # ========================================================================
    # STUDY 3: DISTURBANCE SENSITIVITY (5 scenarios)
    # Variables: disturbance_decel only
    # Fixed: N=40, V=25, s0=2.5, T=1.0
    # ========================================================================
    {
        "nombre": "Disturbance_D08",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -8.0,   # Mild brake
        "enable_visual": False
    },
    {
        "nombre": "Disturbance_D10",
        "num_cars": 40,
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
        "nombre": "Disturbance_D12",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -12.0,  # Standard brake
        "enable_visual": False
    },
    {
        "nombre": "Disturbance_D15",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -15.0,  # Hard brake
        "enable_visual": False
    },
    {
        "nombre": "Disturbance_D18",
        "num_cars": 40,
        "car_max_speed": 25.0,
        "car_acc_max": 2.5,
        "car_dec_max": 4.5,
        "s0": 2.5,
        "t_reaction": 1.0,
        "disturbance_start": 10.0,
        "disturbance_duration": 5.0,
        "disturbance_decel": -18.0,  # Very hard brake
        "enable_visual": False
    },
]

# Total: 14 scenarios (was 7-10 before)
# Each study has controlled variables for clear comparison

# ═════════════════════════════════════════════════════════════════════════
#  CONSTANT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
TRACK_RADIUS = 400                  # meters
CAR_LENGTH = 5.0                    # meters
DT = 0.1                           # simulation timestep (seconds)
MAX_TIME = 300                     # maximum simulation time (seconds)

# Calculate track length
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS

# ═══════════════════════════════════════════════════════════════════════════
#  STATE CLASSIFICATION - GAP-BASED THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════
# These are the key parameters that define traffic states
STOPPED_THRESHOLD = 1.0            # m/s - below this = stopped (was 0.5, too strict!)

# Gap thresholds are velocity-dependent:
# Critical gap = minimum safe distance at current speed
# Formula: gap_critical = S0 + T * v
# Where:
#   S0 = minimum bumper-to-bumper distance (jam spacing)
#   T = time headway (seconds of following distance)
#   v = current velocity

GAP_MULTIPLIER_ATASCO = 1.5        # gap < 1.5 * critical = ATASCO (too close!)
GAP_MULTIPLIER_LIBRE = 2.5         # gap > 2.5 * critical = LIBRE (comfortable)
# Between 1.5x and 2.5x critical = AJUSTANDO (adjusting spacing)


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════
class TrafficSimulator:
    """
    Traffic simulator with gap-based state classification.
    
    States are determined by spacing to leader:
    - PARADO: Not moving (v < 0.5 m/s)
    - ATASCO: Following too close (gap < 1.5 × critical_gap)
    - AJUSTANDO: Adjusting spacing (1.5× < gap < 2.5× critical_gap)
    - LIBRE: Comfortable spacing (gap > 2.5 × critical_gap)
    
    Where critical_gap = S0 + T_reaction × velocity
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
        self.equilibrium_speed = self.car_max_speed * 0.85
        self.equilibrium_gap = TRACK_LENGTH / self.num_cars
        
        # Initialize pygame if visualization enabled
        if self.enable_visual:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption(f"Traffic Simulator - {config['nombre']}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 14)
        
        # Vehicle state arrays - START AT EQUILIBRIUM
        spacing = TRACK_LENGTH / self.num_cars
        self.positions = [i * spacing for i in range(self.num_cars)]
        self.velocities = [self.equilibrium_speed] * self.num_cars
        self.accelerations = [0.0] * self.num_cars
        self.car_states = ["LIBRE"] * self.num_cars
        self.gaps = [spacing - CAR_LENGTH] * self.num_cars  # Track gaps explicitly
        
        # Simulation state
        self.estado = "EQUILIBRIO"
        self.segundo_actual = 0
        self.frame_count = 0
        self.timer_frenada = 0
        self.timer_fin = 0
        
        # Metrics tracking
        self.logs = []
        self.max_stopped = 0
        self.max_en_atasco = 0
        self.max_v_diff = 0
        self.t_dissolve = None
        self.jam_perpetuo = False
        
        # Wave propagation tracking
        self.disturbance_initiated = False
        self.wave_front_car = None
        
        # Energy and efficiency tracking
        self.total_energy_dissipated = 0
        self.cumulative_time_loss = 0
        
        print(f"▶ {config['nombre']}: N={self.num_cars}, V={self.car_max_speed}m/s, S0={self.s0}m, Dist={self.disturbance_decel}m/s²")

    def calculate_critical_gap(self, velocity):
        """
        Calculate the critical gap for a given velocity.
        
        Critical gap = minimum safe following distance at this speed
        Formula: S0 + T_reaction × velocity
        
        This represents the distance needed to:
        - Maintain minimum spacing (S0) when stopped
        - React and brake safely (T_reaction × v) at speed
        """
        return self.s0 + self.t_reaction * velocity

    def classify_car_state(self, i):
        """
        Classify car state based on gap to leader.
        
        The key insight: traffic jams are about SPACING, not speed.
        A car can be going fast but still be "in a jam" if following too close.
        """
        v = self.velocities[i]
        gap = self.gaps[i]
        
        # First check if stopped
        if v < STOPPED_THRESHOLD:
            return "PARADO"
        
        # Calculate what the gap SHOULD be at this speed
        critical_gap = self.calculate_critical_gap(v)
        
        # Classify based on actual gap vs. critical gap
        if gap < GAP_MULTIPLIER_ATASCO * critical_gap:
            # Following too close - this IS a jam condition
            # Even if moving at decent speed!
            return "ATASCO"
        elif gap > GAP_MULTIPLIER_LIBRE * critical_gap:
            # Plenty of space - free flow
            return "LIBRE"
        else:
            # In between - adjusting spacing
            return "AJUSTANDO"

    def calculate_idm_acceleration(self, i):
        """Calculate IDM acceleration for vehicle i."""
        lead = (i + 1) % self.num_cars
        
        # Use pre-calculated gap
        gap = max(0.1, self.gaps[i])
        
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

    def calculate_advanced_metrics(self):
        """Calculate flow, density, efficiency, and wave metrics."""
        # Basic stats
        avg_velocity = sum(self.velocities) / self.num_cars
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        v_diff = max_velocity - min_velocity
        
        # Density (vehicles per km)
        density = (self.num_cars / TRACK_LENGTH) * 1000
        
        # Flow (vehicles per hour)
        flow = density * avg_velocity * 3600 / 1000
        
        # Efficiency
        efficiency = (avg_velocity / self.equilibrium_speed) * 100 if self.equilibrium_speed > 0 else 0
        
        # Time loss
        if self.equilibrium_speed > 0:
            ideal_distance = self.equilibrium_speed * DT
            actual_distance = avg_velocity * DT
            time_loss_per_car = (ideal_distance - actual_distance) / self.equilibrium_speed if actual_distance < ideal_distance else 0
            self.cumulative_time_loss += time_loss_per_car * self.num_cars
        
        # Energy dissipation
        energy_this_step = sum(abs(min(0, a)) * v * DT for a, v in zip(self.accelerations, self.velocities))
        self.total_energy_dissipated += energy_this_step
        
        # Gap statistics (already calculated)
        min_gap = min(self.gaps)
        avg_gap = sum(self.gaps) / len(self.gaps)
        max_gap = max(self.gaps)
        
        # Average critical gap (what gaps SHOULD be)
        avg_critical_gap = sum(self.calculate_critical_gap(v) for v in self.velocities) / self.num_cars
        
        # Gap pressure: how much gaps are compressed below critical
        gap_pressure = avg_critical_gap / avg_gap if avg_gap > 0 else 0
        
        # Wave speed
        wave_speed_kmh = 0
        if self.estado in ["FRENADA_ORIGEN", "ONDA_ACTIVA", "DISOLVIENDO"]:
            slow_cars = [i for i, v in enumerate(self.velocities) if v < self.equilibrium_speed * 0.5]
            if slow_cars and self.wave_front_car is not None:
                wave_distance = (self.positions[self.wave_front_car] - self.positions[slow_cars[-1]]) % TRACK_LENGTH
                time_elapsed = self.segundo_actual - self.disturbance_start
                if time_elapsed > 0:
                    wave_speed_kmh = -(wave_distance / time_elapsed) * 3.6
        
        return {
            'avg_velocity': avg_velocity,
            'min_velocity': min_velocity,
            'max_velocity': max_velocity,
            'v_diff': v_diff,
            'density': density,
            'flow': flow,
            'efficiency': efficiency,
            'gaps': self.gaps,
            'avg_critical_gap': avg_critical_gap,
            'gap_pressure': gap_pressure,
            'energy_dissipated': self.total_energy_dissipated,
            'time_loss': self.cumulative_time_loss,
            'wave_speed_kmh': wave_speed_kmh
        }

    def step_physics(self):
        """Update all vehicle positions and velocities."""
        
        # Calculate accelerations using IDM
        for i in range(self.num_cars):
            self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance
        if self.estado == "FRENADA_ORIGEN":
            self.accelerations[0] = self.disturbance_decel
            if not self.disturbance_initiated:
                self.disturbance_initiated = True
                self.wave_front_car = 0
        
        # Update velocities and positions
        for i in range(self.num_cars):
            self.velocities[i] += self.accelerations[i] * DT
            
            # Realistic stop
            if self.velocities[i] < STOPPED_THRESHOLD and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
            
            # Clamp velocity
            self.velocities[i] = max(0.0, min(self.car_max_speed, self.velocities[i]))
            
            # Update position
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH
        
        # Calculate gaps AFTER all positions updated
        for i in range(self.num_cars):
            lead = (i + 1) % self.num_cars
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            self.gaps[i] = raw_gap - CAR_LENGTH
        
        # Classify states AFTER gaps calculated
        for i in range(self.num_cars):
            self.car_states[i] = self.classify_car_state(i)

    def log_data(self):
        """Record comprehensive statistics."""
        metrics = self.calculate_advanced_metrics()
        
        # State counts
        state_counts = {
            "PARADO": self.car_states.count("PARADO"),
            "ATASCO": self.car_states.count("ATASCO"),
            "AJUSTANDO": self.car_states.count("AJUSTANDO"),
            "LIBRE": self.car_states.count("LIBRE")
        }
        
        stopped = state_counts["PARADO"]
        en_atasco = state_counts["ATASCO"]
        
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        if en_atasco > self.max_en_atasco:
            self.max_en_atasco = en_atasco
        
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
            
            # Vehicle states (GAP-BASED!)
            "coches_parados": state_counts["PARADO"],
            "coches_atasco": state_counts["ATASCO"],
            "coches_ajustando": state_counts["AJUSTANDO"],
            "coches_libre": state_counts["LIBRE"],
            
            # Gap metrics
            "gap_min": round(min(metrics['gaps']), 2),
            "gap_medio": round(sum(metrics['gaps']) / len(metrics['gaps']), 2),
            "gap_max": round(max(metrics['gaps']), 2),
            "gap_critico_medio": round(metrics['avg_critical_gap'], 2),
            "presion_gaps": round(metrics['gap_pressure'], 2),
            
            # Advanced metrics
            "energia_disipada_acum": round(metrics['energy_dissipated'], 1),
            "tiempo_perdido_acum": round(metrics['time_loss'], 1),
            "vel_onda_kmh": round(metrics['wave_speed_kmh'], 1),
            
            # Cumulative extremes
            "max_parados": self.max_stopped,
            "max_en_atasco": self.max_en_atasco,
            "max_v_diff": round(self.max_v_diff, 2)
        })

    def get_state_color(self, state):
        """Get color for gap-based states."""
        colors = {
            "PARADO": (200, 0, 0),        # Red - stopped
            "ATASCO": (255, 140, 0),      # Orange - too close (jam!)
            "AJUSTANDO": (255, 255, 100), # Yellow - adjusting
            "LIBRE": (100, 200, 100),     # Green - comfortable spacing
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
        state_counts = {s: self.car_states.count(s) for s in ["PARADO", "ATASCO", "AJUSTANDO", "LIBRE"]}
        
        info = [
            f"=== {self.config['nombre']} ===",
            f"Time: {self.segundo_actual}s | State: {self.estado}",
            "",
            f"Velocity: {metrics['avg_velocity']:.1f} m/s (min: {metrics['min_velocity']:.1f}, max: {metrics['max_velocity']:.1f})",
            f"V-diff: {metrics['v_diff']:.1f} m/s | Efficiency: {metrics['efficiency']:.0f}%",
            f"Density: {metrics['density']:.1f} veh/km | Flow: {metrics['flow']:.0f} veh/h",
            "",
            f"Gap avg: {metrics['gaps'] and sum(metrics['gaps'])/len(metrics['gaps']):.1f}m | Critical: {metrics['avg_critical_gap']:.1f}m",
            f"Gap pressure: {metrics['gap_pressure']:.2f}× (>1.0 = compressed)",
            "",
            f"🔴 Parados: {state_counts['PARADO']} | 🟠 Atasco: {state_counts['ATASCO']}",
            f"🟡 Ajustando: {state_counts['AJUSTANDO']} | 🟢 Libre: {state_counts['LIBRE']}",
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
            ((200, 0, 0), "Parado (v<0.5)"),
            ((255, 140, 0), "Atasco (gap<1.5×crit)"),
            ((255, 255, 100), "Ajustando (gap 1.5-2.5×)"),
            ((100, 200, 100), "Libre (gap>2.5×crit)"),
        ]
        
        for i, (color, label) in enumerate(legend_items):
            y_pos = 750 + i * 30
            pygame.draw.circle(self.screen, color, (900, y_pos), 8)
            self.screen.blit(self.font.render(label, True, (200, 200, 200)),
                            (920, y_pos - 8))
        
        pygame.display.flip()

    def save_csv(self):
        """Save CSV data."""
        os.makedirs("resultados", exist_ok=True)
        filename = f"resultados/{self.config['nombre']}_data.csv"
        
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
            en_atasco = self.car_states.count("ATASCO")
            
            # State transitions
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
                # Recovery: both v_diff small AND few cars in jam state
                elif v_diff < 2.0 and en_atasco < self.num_cars * 0.1:
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
            print(f"   ✅ t={self.t_dissolve}s | Max parados: {self.max_stopped} | Max atasco: {self.max_en_atasco}")
        elif self.jam_perpetuo:
            print(f"   ❌ Perpetual | Max parados: {self.max_stopped} | Max atasco: {self.max_en_atasco}")
        else:
            print(f"   ⏱️  Timeout at {self.segundo_actual}s")
        
        return "NEXT"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 24 + "TRAFFIC JAM SIMULATOR" + " " * 33 + "║")
    print("║" + " " * 26 + "Gap-Based States" + " " * 36 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\n📊 Running {len(ESCENARIOS)} scenarios...\n")
    
    start_time = time.time()
    
    for i, config in enumerate(ESCENARIOS, 1):
        print(f"[{i}/{len(ESCENARIOS)}] ", end="")
        sim = TrafficSimulator(config)
        result = sim.run()
        
        if result == "QUIT":
            print("\n⚠️  User interrupted simulation")
            break
    
    elapsed = time.time() - start_time
    
    print("\n" + "═" * 80)
    print(f"  ✅ All simulations complete! ({elapsed:.1f}s)")
    print(f"  📁 Results: resultados/")
    print("═" * 80 + "\n")
