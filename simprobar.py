"""
simulador_simple_tuning.py — Simple Traffic Simulation for Parameter Tuning
============================================================================
Single-scenario simulation with only cars (no trucks) for finding optimal
parameters that avoid permanent jams.

EDIT THE PARAMETERS BELOW to experiment and find the sweet spot!

Usage:
    python simulador_simple_tuning.py
"""

import pygame
import math
import csv
import os

# ═══════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS — EDIT THESE TO EXPERIMENT!
# ═══════════════════════════════════════════════════════════════════

# === BASIC SETUP ===
NUM_CARS = 27                        # Number of vehicles (try: 25, 30, 35, 40)
TRACK_RADIUS = 400                   # meters (keep constant)
ENABLE_VISUAL = True                 # Set False for faster testing

# === VEHICLE PROPERTIES ===
CAR_LENGTH = 5.0                     # meters
CAR_MAX_SPEED = 30.0                 # m/s (108 km/h) - try: 25, 30, 35
CAR_ACC_MAX = 2.5                    # m/s² (acceleration) - try: 1.0, 1.5, 2.0
CAR_DEC_MAX = 4.5                    # m/s² (braking) - try: 2.0, 3.0, 4.0

# === IDM PARAMETERS ===
S0 = 2.5                            # minimum gap (m) - try: 3, 5, 7
T_REACTION = 1                    # reaction time (s) - try: 1.0, 1.5, 2.0

# === DISTURBANCE (The sudden brake) ===
DISTURBANCE_START = 10.0            # seconds - when to brake
DISTURBANCE_DURATION = 5.0          # seconds - how long to brake - try: 2, 3, 4, 5
DISTURBANCE_DECEL = -12.0            # m/s² - how hard to brake - try: -6, -8, -10, -12

# === SIMULATION SETTINGS ===
MAX_TIME = 300                      # seconds (cutoff for perpetual jams)
DT = 0.1                           # timestep (don't change)

# Calculate derived values
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS


# ═══════════════════════════════════════════════════════════════════
#  SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════
class SimuladorSimple:
    def __init__(self):
        """Initialize simple simulation with homogeneous cars."""
        if ENABLE_VISUAL:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 950))
            pygame.display.set_caption("Traffic Jam Tuning Tool")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 15)
        
        # Initial positions (evenly spaced)
        spacing = TRACK_LENGTH / NUM_CARS
        self.positions = [i * spacing for i in range(NUM_CARS)]
        self.velocities = [0.0] * NUM_CARS
        self.accelerations = [0.0] * NUM_CARS
        
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
        
        print("\n" + "=" * 70)
        print("  TRAFFIC JAM PARAMETER TUNING")
        print("=" * 70)
        print(f"  Vehicles:     {NUM_CARS}")
        print(f"  Track Length: {TRACK_LENGTH:.0f} m")
        print(f"  Max Speed:    {CAR_MAX_SPEED} m/s")
        print(f"  Acceleration: {CAR_ACC_MAX} m/s²")
        print(f"  Braking:      {CAR_DEC_MAX} m/s²")
        print(f"  Min Gap (S0): {S0} m")
        print(f"  Reaction Time: {T_REACTION} s")
        print(f"  Disturbance:  {DISTURBANCE_DECEL} m/s² for {DISTURBANCE_DURATION}s")
        print("=" * 70 + "\n")

    def calculate_idm_acceleration(self, i):
        """Calculate IDM acceleration for vehicle i."""
        lead = (i + 1) % NUM_CARS
        
        # Calculate gap
        raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
        gap = max(0.1, raw_gap - CAR_LENGTH)
        
        v = self.velocities[i]
        dv = v - self.velocities[lead]
        
        # Free flow term
        if CAR_MAX_SPEED > 0:
            v_ratio = v / CAR_MAX_SPEED
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
        """Update all vehicle positions and velocities.
        if self.estado == "ESTABILIZANDO":
            # Gentle acceleration to cruising speed
            for i in range(NUM_CARS):
                if self.velocities[i] < CAR_MAX_SPEED:
                    self.accelerations[i] = CAR_ACC_MAX * 0.5
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
            self.velocities[i] = max(0.0, min(CAR_MAX_SPEED, self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH
            """"""
        # Update velocities and positions
        for i in range(NUM_CARS):
            self.velocities[i] += self.accelerations[i] * DT
            
            # 🛑 REALISTIC STOP: If moving very slowly and still braking, force a dead stop
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
                self.accelerations[i] = 0.0
                
            self.velocities[i] = max(0.0, min(CAR_MAX_SPEED, self.velocities[i]))
        
"""
        if self.estado == "ESTABILIZANDO":
            # Gentle acceleration to cruising speed
            for i in range(NUM_CARS):
                if self.velocities[i] < CAR_MAX_SPEED:
                    self.accelerations[i] = CAR_ACC_MAX * 0.5
                else:
                    self.accelerations[i] = 0.0
        else:
            # IDM control
            for i in range(NUM_CARS):
                self.accelerations[i] = self.calculate_idm_acceleration(i)
        
        # Apply disturbance
        if self.estado == "FRENADA_ORIGEN":
            self.accelerations[0] = DISTURBANCE_DECEL
        
        # Update velocities and positions
        for i in range(NUM_CARS):
            self.velocities[i] += self.accelerations[i] * DT
            
            # 🛑 REALISTIC STOP: If moving very slowly and still braking, force a dead stop
            if self.velocities[i] < 0.5 and self.accelerations[i] < 0:
                self.velocities[i] = 0.0
                self.accelerations[i] = 0.0
                
            self.velocities[i] = max(0.0, min(CAR_MAX_SPEED, self.velocities[i]))
            self.positions[i] += self.velocities[i] * DT
            self.positions[i] %= TRACK_LENGTH


    def log_data(self):
        """Record statistics for this second."""
        avg_velocity = sum(self.velocities) / NUM_CARS
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        
        # Calculate speed variance (v_diff)
        v_diff = max_velocity - min_velocity
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        cruising = sum(1 for v in self.velocities if v >= CAR_MAX_SPEED - 1.0)
        braking = sum(1 for a in self.accelerations if a < -1.0)
        
        # Track maximum stopped
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        # Calculate gaps
        gaps = []
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
            gaps.append(gap)
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado": self.estado,
            "avg_velocity": round(avg_velocity, 2),
            "min_velocity": round(min_velocity, 2),
            "max_velocity": round(max_velocity, 2),
            "v_diff": round(v_diff, 2),          # <-- New Variance Metric
            "vehicles_cruising": cruising,
            "vehicles_braking": braking,
            "vehicles_stopped": stopped,
            "min_gap": round(min(gaps), 2),
            "avg_gap": round(sum(gaps) / len(gaps), 2),
        })



        """   
        #Record statistics for this second.
        avg_velocity = sum(self.velocities) / NUM_CARS
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)
        
        stopped = sum(1 for v in self.velocities if v < 1.0)
        cruising = sum(1 for v in self.velocities if v >= CAR_MAX_SPEED - 1.0)
        braking = sum(1 for a in self.accelerations if a < -1.0)
        
        # Track maximum stopped
        if stopped > self.max_stopped:
            self.max_stopped = stopped
        
        # Calculate gaps
        gaps = []
        for i in range(NUM_CARS):
            lead = (i + 1) % NUM_CARS
            raw_gap = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
            gap = raw_gap - CAR_LENGTH
            gaps.append(gap)
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado": self.estado,
            "avg_velocity": round(avg_velocity, 2),
            "min_velocity": round(min_velocity, 2),
            "max_velocity": round(max_velocity, 2),
            "vehicles_cruising": cruising,
            "vehicles_braking": braking,
            "vehicles_stopped": stopped,
            "min_gap": round(min(gaps), 2),
            "avg_gap": round(sum(gaps) / len(gaps), 2),
        })
        """
    def get_vehicle_color(self, i):
        """Calculate vehicle color based on acceleration."""
        accel = self.accelerations[i]
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
        if not ENABLE_VISUAL:
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
            
            # White outline for disturbance source
            if i == 0:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(x), int(y)), 10)
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
        
        # Statistics panel
        avg_vel = sum(self.velocities) / NUM_CARS
        stopped = sum(1 for v in self.velocities if v < 1.0)
        
        info = [
            "=== TRAFFIC JAM TUNING ===",
            f"Time: {self.segundo_actual}s   State: {self.estado}",
            "",
            "=== PARAMETERS ===",
            f"Vehicles:     {NUM_CARS}",
            f"Max Speed:    {CAR_MAX_SPEED} m/s",
            f"Acceleration: {CAR_ACC_MAX} m/s²",
            f"Braking:      {CAR_DEC_MAX} m/s²",
            f"Min Gap (S0): {S0} m",
            f"Reaction:     {T_REACTION} s",
            f"Disturbance:  {DISTURBANCE_DECEL} m/s² × {DISTURBANCE_DURATION}s",
            "",
            "=== CURRENT STATE ===",
            f"Avg velocity:  {avg_vel:.2f} m/s",
            f"Min velocity:  {min(self.velocities):.2f} m/s",
            f"Max velocity:  {max(self.velocities):.2f} m/s",
            "",
            f"Cruising:      {sum(1 for v in self.velocities if v >= CAR_MAX_SPEED - 1.0)}",
            f"Braking:       {sum(1 for a in self.accelerations if a < -1.0)}",
            f"Stopped:       {stopped}",
            f"Max stopped:   {self.max_stopped}",
            "",
            "=== RESULTS ===",
        ]
        
        if self.t_dissolve:
            info.append(f"✅ Jam dissolved: {self.t_dissolve}s")
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
        legend_x, legend_y = 900, 800
        legend_items = [
            ((255, 0, 0), "Hard braking", 8),
            ((255, 255, 0), "Coasting", 8),
            ((127, 255, 0), "Accelerating", 8),
        ]
        
        for i, (color, label, size) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            pygame.draw.circle(self.screen, color, (legend_x, y_pos), size)
            self.screen.blit(self.font.render(label, True, (200, 200, 200)),
                            (legend_x + 20, y_pos - 8))
        
        pygame.display.flip()

    def save_csv(self):
        """Save time-series data to CSV."""
        os.makedirs("simprobar", exist_ok=True)
        
        # Create filename based on key parameters
        filename = (f"simprobar/N{NUM_CARS}_V{int(CAR_MAX_SPEED)}_"
                   f"D{abs(int(DISTURBANCE_DECEL))}_T{int(DISTURBANCE_DURATION)}.csv")
        
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
            if ENABLE_VISUAL:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # Calculate key metrics for state transitions
            v_min = min(self.velocities)
            v_max = max(self.velocities)
            v_diff = v_max - v_min  # The new metric for stability!

            # 1. ESTABILIZANDO -> FLUJO_SINCRO (Synchronized Flow)
            if self.estado == "ESTABILIZANDO":
                # System is stable when all cars are going roughly the same speed
                if v_diff < 1.0 and self.segundo_actual > 5: 
                    self.estado = "FLUJO_SINCRO"
                    equilibrium_speed = sum(self.velocities) / NUM_CARS
                    print(f"[{self.segundo_actual}s] 🟢 STABLE FLOW ACHIEVED at {equilibrium_speed:.1f} m/s")

            # 2. FLUJO_SINCRO -> FRENADA_ORIGEN (The Disturbance)
            if self.estado == "FLUJO_SINCRO" and self.segundo_actual >= DISTURBANCE_START:
                self.estado = "FRENADA_ORIGEN"
                self.timer_frenada = DISTURBANCE_DURATION
                print(f"[{self.segundo_actual}s] 🚨 DISTURBANCE STARTED")

            # 3. FRENADA_ORIGEN -> ONDA_ACTIVA
            if self.estado == "FRENADA_ORIGEN":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "ONDA_ACTIVA"
                    print(f"[{self.segundo_actual}s] 💥 SHOCKWAVE PROPAGATING")

            # 4 & 5. Tracking the Wave (ONDA_ACTIVA <--> DISOLVIENDO)
            if self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                # If there is a hard stop (someone going < 1m/s)
                if v_min < 1.0 and self.estado == "DISOLVIENDO":
                    self.estado = "ONDA_ACTIVA"
                    print(f"[{self.segundo_actual}s] ⚠️ WAVE RE-FORMED (Cars stopped again)")
                
                # If no one is fully stopped, but the wave is still causing accordion effect
                elif v_min >= 1.0 and self.estado == "ONDA_ACTIVA":
                    self.estado = "DISOLVIENDO"
                    print(f"[{self.segundo_actual}s] 📉 QUEUE CLEARED - Accordion phase started")

                # 6. DISOLVIENDO -> RECUPERADO (End of Jam)
                # The jam is dead when the speed difference between fastest/slowest car is small again
                if self.estado == "DISOLVIENDO" and v_diff < 2.0:
                    if self.t_dissolve is None:
                        self.estado = "RECUPERADO"
                        self.t_dissolve = self.segundo_actual
                        final_speed = sum(self.velocities) / NUM_CARS
                        print(f"[{self.segundo_actual}s] ✅ JAM DISSOLVED! Traffic stabilized at {final_speed:.1f} m/s")
                        self.timer_fin = 5

            # Wait before finishing
            if self.estado == "RECUPERADO":
                self.timer_fin -= DT
                if self.timer_fin <= 0:
                    running = False
                    
            # Perpetual jam detection (If it's been active too long)
            if self.segundo_actual > 200 and self.estado in ["ONDA_ACTIVA", "DISOLVIENDO"]:
                stopped_cars = sum(1 for v in self.velocities if v < 1.0)
                if stopped_cars > NUM_CARS * 0.2:  # More than 20% still stopped
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
            if ENABLE_VISUAL:
                self.draw()
                self.clock.tick(30)
        
        # Save results
        self.save_csv()
        
        # Print summary
        print("\n" + "=" * 70)
        print("  SIMULATION COMPLETE")
        print("=" * 70)
        if self.t_dissolve:
            print(f"  ✅ SUCCESS: Jam dissolved in {self.t_dissolve}s")
            print(f"  Max stopped vehicles: {self.max_stopped}")
        elif self.jam_perpetuo:
            print(f"  ❌ FAILURE: Perpetual jam (max {self.max_stopped} vehicles stopped)")
            print(f"  Try: Reduce NUM_CARS, increase S0, or soften DISTURBANCE_DECEL")
        else:
            print(f"  ⏱️  Simulation ended at {self.segundo_actual}s")
        print("=" * 70 + "\n")
        
        return "DONE"
    
        """Main simulation loop.
        running = True
        
        while running and self.segundo_actual < MAX_TIME:
            # Handle pygame events
            if ENABLE_VISUAL:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "QUIT"
            
            # State transitions
            if self.estado == "ESTABILIZANDO" and self.segundo_actual >= DISTURBANCE_START:
                self.estado = "FRENANDO"
                self.timer_frenada = DISTURBANCE_DURATION
                print(f"[{self.segundo_actual}s] 🚨 DISTURBANCE STARTED")
            
            if self.estado == "FRENANDO":
                self.timer_frenada -= DT
                if self.timer_frenada <= 0:
                    self.estado = "RECUPERANDO"
                    print(f"[{self.segundo_actual}s] 🔄 RECOVERY PHASE")
            
            # Physics update
            self.step_physics()
            
            # Data logging (every second)
            self.frame_count += 1
            if self.frame_count >= 10:  # 10 frames * 0.1s = 1 second
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                # Check for jam dissolution
                if self.estado == "RECUPERANDO":
                    all_at_target = all(v >= CAR_MAX_SPEED - 1.0 for v in self.velocities)
                    if all_at_target and self.t_dissolve is None:
                        self.t_dissolve = self.segundo_actual
                        print(f"[{self.segundo_actual}s] ✅ JAM DISSOLVED!")
                        self.estado = "FIN_ESPERA"
                        self.timer_fin = 5
                
                # Wait before finishing
                if self.estado == "FIN_ESPERA":
                    self.timer_fin -= 1
                    if self.timer_fin <= 0:
                        running = False
                
                # Perpetual jam detection
                if self.segundo_actual > 200 and self.estado == "RECUPERANDO":
                    stopped = sum(1 for v in self.velocities if v < 1.0)
                    if stopped > NUM_CARS * 0.3:  # More than 30% still stopped
                        print(f"[{self.segundo_actual}s] ❌ PERPETUAL JAM DETECTED!")
                        self.jam_perpetuo = True
                        running = False
            
            # Visualization
            if ENABLE_VISUAL:
                self.draw()
                self.clock.tick(30)
        
        # Save results
        self.save_csv()
        
        # Print summary
        print("\n" + "=" * 70)
        print("  SIMULATION COMPLETE")
        print("=" * 70)
        if self.t_dissolve:
            print(f"  ✅ SUCCESS: Jam dissolved in {self.t_dissolve}s")
            print(f"  Max stopped vehicles: {self.max_stopped}")
        elif self.jam_perpetuo:
            print(f"  ❌ FAILURE: Perpetual jam (max {self.max_stopped} vehicles stopped)")
            print(f"  Try: Reduce NUM_CARS, increase S0, or soften DISTURBANCE_DECEL")
        else:
            print(f"  ⏱️  Simulation ended at {self.segundo_actual}s")
        print("=" * 70 + "\n")
        
        return "DONE"
        """


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sim = SimuladorSimple()
    result = sim.run()
    
    if ENABLE_VISUAL:
        pygame.quit()
    
    print("\n💡 TIP: Edit the parameters at the top of this file and run again!")
    print("   Try different values for NUM_CARS, DISTURBANCE_DECEL, S0, etc.\n")
