import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Configuración de estilo para que parezcan gráficas de un "paper" científico
plt.style.use('bmh')

def grafica_onda_choque(archivo_csv):
    """
    Gráfica 1: Área Apilada. 
    Muestra cómo el tráfico pasa de crucero -> frenando -> atasco -> recuperando.
    """
    df = pd.read_csv(archivo_csv)
    nombre_base = archivo_csv.replace('_datos.csv', '')
    
    df_estados = df[['segundo', 'coches_crucero', 'coches_frenando', 'coches_atasco', 'coches_recuperando']]
    df_estados.set_index('segundo', inplace=True)
    
    # --- EL FIX ESTÁ AQUÍ ---
    # Renombramos las columnas directamente en los datos
    df_estados.columns = ['Crucero', 'Frenando (Onda de choque)', 'Atasco (Núcleo)', 'Recuperando (Onda de descarga)']
    
    colores = ['#32CD32', '#FFA500', '#FF3232', '#32C8FF'] 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_estados.plot.area(ax=ax, color=colores, alpha=0.8)
    
    ax.set_title(f"Evolución de la Onda de Tráfico - Escenario: {nombre_base}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tiempo de simulación (Segundos)", fontsize=12)
    ax.set_ylabel("Número de vehículos", fontsize=12)
    
    # Ahora solo le decimos dónde poner la leyenda, los nombres y colores los saca solos
    ax.legend(loc='upper right') 
    # ------------------------

    ax.set_ylim(0, df['coches_crucero'].max() + 2)
    
    nombre_imagen = f"{nombre_base}_Grafica_Areas.png"
    plt.tight_layout()
    plt.savefig(nombre_imagen, dpi=300)
    plt.close()
    print(f"✅ Generada: {nombre_imagen}")

def grafica_comparativa_frenada(csv_5s, csv_10s, vel_base):
    """
    Gráfica 2: Líneas Comparativas.
    Compara la caída de velocidad media entre un frenazo corto (5s) y uno largo (10s).
    """
    if not os.path.exists(csv_5s) or not os.path.exists(csv_10s):
        return # Si no encuentra los dos archivos, se salta esta gráfica
        
    df5 = pd.read_csv(csv_5s)
    df10 = pd.read_csv(csv_10s)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df5['segundo'], df5['vel_media_sistema'], label='Frenazo de 5s', color='blue', linewidth=2)
    ax.plot(df10['segundo'], df10['vel_media_sistema'], label='Frenazo de 10s', color='red', linewidth=2, linestyle='--')
    
    ax.set_title(f"Impacto del Tiempo de Frenada en la Velocidad Media (V0 = {vel_base} m/s)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tiempo de simulación (Segundos)", fontsize=12)
    ax.set_ylabel("Velocidad Media del Sistema (m/s)", fontsize=12)
    
    # Línea de referencia (Velocidad de crucero)
    ax.axhline(y=vel_base, color='green', linestyle=':', label='Velocidad Ideal (Crucero)')
    
    ax.legend(loc='lower right')
    
    nombre_imagen = f"Comparativa_V{vel_base}_F5_vs_F10.png"
    plt.tight_layout()
    plt.savefig(nombre_imagen, dpi=300)
    plt.close()
    print(f"✅ Generada: {nombre_imagen}")

# ==========================================
# MOTOR PRINCIPAL DE EJECUCIÓN
# ==========================================
print("Iniciando análisis de datos...")

# 1. Encontrar todos los CSV en la carpeta actual
archivos_csv = glob.glob("Exp_*_datos.csv")

if not archivos_csv:
    print("😡 No se encontraron archivos .csv. Asegúrate de ejecutar primero la simulación.")
else:
    # 2. Generar gráfica de áreas para cada archivo individual
    for archivo in archivos_csv:
        grafica_onda_choque(archivo)
        
    # 3. Generar gráficas comparativas (buscando pares automáticos)
    # Lista de las velocidades que pusiste en el simulador
    velocidades = [15, 20, 25, 30]
    
    for v in velocidades:
        file_5s = f"Exp_V{v}_F5_datos.csv"
        file_10s = f"Exp_V{v}_F10_datos.csv"
        grafica_comparativa_frenada(file_5s, file_10s, v)

print("\n✅ ¡Análisis completado! Revisa las imágenes .png en tu carpeta.")