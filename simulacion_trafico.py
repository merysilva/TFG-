import pygame
import math
import csv
import time
import os

# --- CONFIGURACIÓN DE EXPERIMENTOS ---
ESCENARIOS = [
    {"nombre": "Exp_V15_F5", "v_crucero": 15.0, "t_frenada": 5.0},
    {"nombre": "Exp_V20_F5", "v_crucero": 20.0, "t_frenada": 5.0},
    {"nombre": "Exp_V25_F5", "v_crucero": 25.0, "t_frenada": 5.0},
    {"nombre": "Exp_V30_F5", "v_crucero": 30.0, "t_frenada": 5.0},
    {"nombre": "Exp_V15_F10", "v_crucero": 15.0, "t_frenada": 10.0},
    {"nombre": "Exp_V20_F10", "v_crucero": 20.0, "t_frenada": 10.0},
    {"nombre": "Exp_V25_F10", "v_crucero": 25.0, "t_frenada": 10.0},
    {"nombre": "Exp_V30_F10", "v_crucero": 30.0, "t_frenada": 10.0}
]

# --- PARÁMETROS FÍSICOS CONSTANTES ---
NUM_CARS = 38
TRACK_RADIUS = 400  
TRACK_LENGTH = 2 * math.pi * TRACK_RADIUS
T_REACCION = 1.5
S0 = 5.0
ACC_MAX = 1.2
DEC_MAX = 2.0

class SimuladorTFG:
    def __init__(self, config):
        pygame.init()
        self.screen = pygame.display.set_mode((900, 900)) 
        pygame.display.set_caption(f"TFG - Ejecutando: {config['nombre']}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)
        
        self.config = config
        self.v0 = config['v_crucero']
        self.duracion_frenada = config['t_frenada']
        
        self.positions = [i * (TRACK_LENGTH / NUM_CARS) for i in range(NUM_CARS)]
        self.velocities = [self.v0 for _ in range(NUM_CARS)]
        self.car_states = ["CRUCERO"] * NUM_CARS # Guardará el estado textual de cada coche
        
        self.logs = []
        self.frame_count = 0
        self.segundo_actual = 0
        self.estado = "ESTABILIZANDO" 
        self.timer_frenada = 0

    def guardar_csv(self):
        filename = f"{self.config['nombre']}_datos.csv"
        keys = self.logs[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.logs)
        print(f" 🤑 Archivo {filename} generado con éxito.")

    def log_data(self):
        # Contamos cuántos coches hay en cada estado
        counts = {
            "CRUCERO": self.car_states.count("CRUCERO"),
            "FRENANDO": self.car_states.count("FRENANDO"),
            "ATASCO": self.car_states.count("ATASCO"),
            "RECUPERANDO": self.car_states.count("RECUPERANDO")
        }
        
        vel_media_sistema = sum(self.velocities) / NUM_CARS
        
        self.logs.append({
            "segundo": self.segundo_actual,
            "estado_experimento": self.estado,
            "coches_crucero": counts["CRUCERO"],
            "coches_frenando": counts["FRENANDO"],
            "coches_atasco": counts["ATASCO"],
            "coches_recuperando": counts["RECUPERANDO"],
            "vel_media_sistema": round(vel_media_sistema, 2),
            "vel_min": round(min(self.velocities), 2)
        })

    def run(self):
        running = True
        while running:
            dt = 0.1
            self.screen.fill((20, 20, 25))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return "QUIT"

            if self.estado == "ESTABILIZANDO" and self.segundo_actual >= 7:
                self.estado = "FRENANDO"
                self.timer_frenada = self.duracion_frenada
            
            accelerations = [0] * NUM_CARS
            
            # --- CÁLCULO DE FÍSICA ---
            if self.estado == "ESTABILIZANDO":
                for i in range(NUM_CARS):
                    self.velocities[i] = self.v0
                    accelerations[i] = 0
            else:
                for i in range(NUM_CARS):
                    lead = (i + 1) % NUM_CARS
                    s = (self.positions[lead] - self.positions[i]) % TRACK_LENGTH
                    s = max(0.1, s - 5.0)
                    
                    v = self.velocities[i]
                    dv = v - self.velocities[lead]
                    
                    s_star = S0 + max(0, v * T_REACCION + (v * dv) / (2 * math.sqrt(ACC_MAX * DEC_MAX)))
                    accelerations[i] = ACC_MAX * (1 - (v / self.v0)**4 - (s_star / s)**2)

            if self.estado == "FRENANDO":
                accelerations[0] = -12.0
                self.timer_frenada -= dt
                if self.timer_frenada <= 0:
                    self.estado = "RECUPERANDO"

            # --- ACTUALIZACIÓN DE POSICIÓN Y ESTADO (Tu nueva lógica) ---
            for i in range(NUM_CARS):
                # 1. Actualizar movimiento
                self.velocities[i] = max(0, self.velocities[i] + accelerations[i] * dt)
                self.positions[i] = (self.positions[i] + self.velocities[i] * dt) % TRACK_LENGTH
                
                # 2. Clasificar el estado del coche
                v = self.velocities[i]
                a = accelerations[i]
                
                if v >= self.v0 - 1.0:
                    self.car_states[i] = "CRUCERO"
                elif v < self.v0 / 2:
                    self.car_states[i] = "ATASCO"
                else:
                    # Zona media: Usamos la aceleración para saber si entra o sale
                    if a < 0:
                        self.car_states[i] = "FRENANDO"
                    else:
                        self.car_states[i] = "RECUPERANDO"

            # --- RECOLECCIÓN DE DATOS ---
            self.frame_count += 1
            if self.frame_count >= 10: 
                self.segundo_actual += 1
                self.log_data()
                self.frame_count = 0
                
                if self.estado == "RECUPERANDO":
                    # 1. ¿Se ha disipado el atasco? (Todos en verde)
                    if self.car_states.count("CRUCERO") == NUM_CARS:
                        self.estado = "FIN_ESPERA" # Nuevo estado intermedio
                        self.timer_fin = 5 # Esperar 5 segundos antes de cerrar
                    
                    # 2. Cortafuegos: El atasco es perpetuo y llevamos 5 minutos de simulación
                    elif self.segundo_actual > 400: 
                        print(f"[{self.config['nombre']}] 😡 Atasco perpetuo detectado. Abortando por tiempo.")
                        self.estado = "FIN"
                        running = False
                        
                elif self.estado == "FIN_ESPERA":
                    # Mantiene la simulación corriendo en verde 5 segundos para que lo veas
                    self.timer_fin -= 1
                    if self.timer_fin <= 0:
                        self.estado = "FIN"
                        running = False

            # --- DIBUJAR PISTA Y COCHES ---
            pygame.draw.circle(self.screen, (50, 50, 50), (450, 450), TRACK_RADIUS, 30)
            
            # Diccionario de colores para los 4 estados
            colores_estado = {
                "CRUCERO": (50, 255, 50),      # Verde
                "FRENANDO": (255, 165, 0),     # Naranja
                "ATASCO": (255, 50, 50),       # Rojo
                "RECUPERANDO": (50, 200, 255)  # Azul claro
            }
            
            for i in range(NUM_CARS):
                angle = (self.positions[i] / TRACK_LENGTH) * 2 * math.pi
                x, y = 450 + TRACK_RADIUS * math.cos(angle), 450 + TRACK_RADIUS * math.sin(angle)
                
                color = colores_estado[self.car_states[i]]
                radius = 12 if i == 0 else 8
                pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

            # --- UI (Panel de Control) ---
            info = [
                f"Escenario: {self.config['nombre']}",
                f"Estado General: {self.estado}",
                f"Tiempo: {self.segundo_actual}s",
                f"--- DISTRIBUCIÓN DEL TRÁFICO ---",
                f"Crucero (Verde):     {self.car_states.count('CRUCERO')}",
                f"Frenando (Naranja):  {self.car_states.count('FRENANDO')}",
                f"Atasco (Rojo):       {self.car_states.count('ATASCO')}",
                f"Recuperando (Azul):  {self.car_states.count('RECUPERANDO')}"
            ]
            for idx, text in enumerate(info):
                surf = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(surf, (20, 20 + idx * 25))

            pygame.display.flip()
            self.clock.tick(60)

        self.guardar_csv()
        return "NEXT"

# EJECUTAR EXPERIMENTOS
for config in ESCENARIOS:
    sim = SimuladorTFG(config)
    resultado = sim.run()
    if resultado == "QUIT": break

pygame.quit()