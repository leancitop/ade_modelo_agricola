"""
Script para generar gráfico de evolución temporal del NDVI agrupado por tipo de uso del suelo
para Coronel Suárez: Cultivos vs No agrícola (agrupando NAs, 255 y No agrícola)
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "proc")
INTA_DIR = os.path.join(RAW_DIR, "INTA_23_24")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "img")

OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Archivos
RASTER_PATH = os.path.join(PROC_DIR, "11_NDVI_inta_verano_coronel_suarez.tif")

print("=" * 80)
print("EVOLUCION TEMPORAL NDVI AGRUPADO - CORONEL SUAREZ")
print("=" * 80)

# Meses correspondientes a las bandas 6-12 (NDVI temporal desde diciembre 2023)
meses = [
    "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"
]

# Definir grupos
# Cultivos: todos los valores de cultivos de verano (excluyendo barbecho, no agrícola, máscara)
cultivos_ver = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 28, 30]  # Todos los cultivos de verano

# No agrícola agrupado: No agrícola (22), NAs, 255, y otros valores especiales
no_agricola_agrupado = [22, 255, 0, 25, 31]  # No agrícola, máscara, y valores especiales

print(f"\nGrupos definidos:")
print(f"  Cultivos: {cultivos_ver}")
print(f"  No agrícola (agrupado): {no_agricola_agrupado}")

# Leer el raster
print(f"\nLeyendo raster: {RASTER_PATH}")
with rasterio.open(RASTER_PATH) as src:
    # Leer banda 1 (inta verano)
    banda_verano = src.read(1).astype(np.float32)
    
    # Leer bandas 6-12 (NDVI temporal desde diciembre 2023)
    bandas_ndvi = []
    for i in range(6, 13):  # Bandas 6 a 12
        banda = src.read(i)
        if src.nodata is not None:
            banda = banda.astype(np.float32)
            banda[banda == src.nodata] = np.nan
        else:
            banda = banda.astype(np.float32)
        bandas_ndvi.append(banda)
    
    print(f"  Dimensiones: {src.height} x {src.width}")
    print(f"  Bandas NDVI leídas: {len(bandas_ndvi)}")

# Verificar qué categorías hay en el recorte
valores_unicos = np.unique(banda_verano[~np.isnan(banda_verano)])
print(f"\nValores únicos en el recorte: {sorted(valores_unicos.astype(int).tolist())}")

# Verificar si hay barbecho (21)
tiene_barbecho = 21 in valores_unicos
print(f"  ¿Hay barbecho (21)? {tiene_barbecho}")

# Función para calcular NDVI promedio por grupo
def calcular_ndvi_por_grupo(banda_categorias, bandas_ndvi, grupos):
    resultados = defaultdict(list)
    
    for idx_mes, banda_ndvi in enumerate(bandas_ndvi):
        for grupo_nombre, categorias_grupo in grupos.items():
            # Crear máscara para todas las categorías del grupo
            mask_grupo = np.zeros_like(banda_categorias, dtype=bool)
            
            if grupo_nombre == "Cultivos":
                # Para cultivos: incluir solo valores de cultivos
                for cat_val in categorias_grupo:
                    mask_grupo |= (banda_categorias == cat_val)
            elif grupo_nombre == "No agrícola":
                # Para no agrícola: incluir valores específicos Y también NaN
                for cat_val in categorias_grupo:
                    mask_grupo |= (banda_categorias == cat_val)
                # También incluir píxeles con NaN en la banda de categorías
                mask_grupo |= np.isnan(banda_categorias)
            
            # Máscara para valores válidos de NDVI
            mask_valido = ~np.isnan(banda_ndvi)
            mask_final = mask_grupo & mask_valido
            
            if np.sum(mask_final) > 0:
                ndvi_promedio = np.nanmean(banda_ndvi[mask_final])
            else:
                ndvi_promedio = np.nan
            
            resultados[grupo_nombre].append(ndvi_promedio)
    
    return resultados

# Calcular para VERANO
grupos_ver = {
    'Cultivos': cultivos_ver,
    'No agrícola': no_agricola_agrupado
}

print(f"\nCalculando NDVI promedio por grupo...")
ndvi_por_grupo_ver = calcular_ndvi_por_grupo(banda_verano, bandas_ndvi, grupos_ver)

# Mostrar estadísticas
print(f"\nEstadísticas por grupo:")
for grupo_nombre, valores_ndvi in ndvi_por_grupo_ver.items():
    valores_validos = [v for v in valores_ndvi if not np.isnan(v)]
    if valores_validos:
        print(f"  {grupo_nombre}:")
        print(f"    Min: {min(valores_validos):.3f}")
        print(f"    Max: {max(valores_validos):.3f}")
        print(f"    Promedio: {np.mean(valores_validos):.3f}")

# Crear visualización
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# Colores para los grupos
colores_grupos = {
    'Cultivos': '#42f4ce',  # Verde azulado (cyan)
    'No agrícola': '#e6f0c2'  # Beige claro
}

# Graficar cada grupo
for grupo_nombre in ['Cultivos', 'No agrícola']:
    valores_ndvi = ndvi_por_grupo_ver[grupo_nombre]
    color = colores_grupos[grupo_nombre]
    
    if any(not np.isnan(v) for v in valores_ndvi):
        ax.plot(meses, valores_ndvi, marker='o', label=grupo_nombre, 
               color=color, linewidth=2.5, markersize=8)

ax.set_xticks(range(len(meses)))
ax.set_xticklabels(meses, rotation=45, ha='right')
ax.set_xlabel("Mes", fontsize=12)
ax.set_ylabel("NDVI promedio", fontsize=12)
ax.set_title("Evolución temporal NDVI por grupo - VERANO 2024 - Coronel Suárez", 
            fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11, loc='best')
ax.set_ylim(bottom=0)

plt.suptitle("Evolución temporal del NDVI agrupado por tipo de uso del suelo - Coronel Suárez\n(Cultivos vs No agrícola agrupado)", 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Guardar figura
output_path = OUTPUT_DIR / "evolucion_temporal_ndvi_coronel_suarez.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Visualización guardada en: {output_path}")

plt.show()

print(f"\nGrafico generado exitosamente")

