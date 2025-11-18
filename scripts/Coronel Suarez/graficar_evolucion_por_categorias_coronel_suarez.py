"""
Script para generar gráfico de evolución temporal del NDVI por categorías individuales del INTA
para Coronel Suárez (solo verano, desde diciembre 2023)
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from collections import defaultdict
import xml.etree.ElementTree as ET
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
QML_PATH = os.path.join(INTA_DIR, "MNC_ver24.qml")

print("=" * 80)
print("EVOLUCION TEMPORAL NDVI POR CATEGORIAS - CORONEL SUAREZ")
print("=" * 80)

# Meses correspondientes a las bandas 6-12 (NDVI temporal desde diciembre 2023)
meses = [
    "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"
]

# Función para parsear QML y obtener categorías con colores
def parsear_categorias_qml(qml_path):
    tree = ET.parse(qml_path)
    root = tree.getroot()
    
    categorias = {}
    labels = []
    vals = []
    hex_colors = []
    
    # Buscar en el namespace de QGIS
    namespaces = {'qgis': 'http://www.qgis.org/schema/3.0'}
    
    colorrampshader = root.find('.//qgis:colorrampshader', namespaces)
    if colorrampshader is None:
        # Intentar sin namespace
        colorrampshader = root.find('.//colorrampshader')
    
    if colorrampshader is not None:
        items = colorrampshader.findall('.//item')
        for item in items:
            value = int(float(item.get('value')))
            color_hex = item.get('color', '#ffffff')
            label = item.get('label', str(value))
            
            categorias[value] = {
                'label': label,
                'color': color_hex
            }
            labels.append(label)
            vals.append(value)
            hex_colors.append(color_hex)
    
    return categorias, labels, vals, hex_colors

# Parsear categorías del QML
print(f"\nLeyendo archivo QML: {QML_PATH}")
categorias_qml, labels_ver, vals_ver, hex_colors_ver = parsear_categorias_qml(QML_PATH)
print(f"  Categorías encontradas: {len(categorias_qml)}")

# Valores a excluir
valores_excluir = [255, 0, 25, 31]

# Leer el raster
print(f"\nLeyendo raster: {RASTER_PATH}")
with rasterio.open(RASTER_PATH) as src:
    # Leer banda 1 (inta verano)
    banda_verano = src.read(1)
    
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

# Identificar categorías presentes en VERANO
categorias_presentes_ver = {}
for cat_val in vals_ver:
    if cat_val not in valores_excluir:
        mask = (banda_verano == cat_val)
        n_pixeles = np.sum(mask)
        if n_pixeles > 0:
            idx = vals_ver.index(cat_val)
            categorias_presentes_ver[cat_val] = {
                'label': labels_ver[idx],
                'color': hex_colors_ver[idx],
                'n_pixeles': n_pixeles
            }

print(f"\nCategorías presentes en VERANO: {len(categorias_presentes_ver)}")
for cat_val, info in sorted(categorias_presentes_ver.items()):
    print(f"  {info['label']} (valor={cat_val}, píxeles={info['n_pixeles']:,})")

# Función para calcular NDVI promedio por categoría individual
def calcular_ndvi_por_categoria(banda_categorias, bandas_ndvi, categorias_presentes):
    resultados = defaultdict(list)
    
    for idx_mes, banda_ndvi in enumerate(bandas_ndvi):
        for cat_val, info in categorias_presentes.items():
            # Máscara: píxeles de esta categoría Y válidos en NDVI
            mask_categoria = (banda_categorias == cat_val)
            mask_valido = ~np.isnan(banda_ndvi)
            mask_final = mask_categoria & mask_valido
            
            if np.sum(mask_final) > 0:
                ndvi_promedio = np.nanmean(banda_ndvi[mask_final])
            else:
                ndvi_promedio = np.nan
            
            resultados[cat_val].append(ndvi_promedio)
    
    return resultados

# Calcular NDVI por categoría para VERANO
ndvi_por_categoria_ver = calcular_ndvi_por_categoria(banda_verano, bandas_ndvi, categorias_presentes_ver)

# Crear visualización
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# Graficar cada categoría de verano
for cat_val in sorted(categorias_presentes_ver.keys()):
    valores_ndvi = ndvi_por_categoria_ver[cat_val]
    info = categorias_presentes_ver[cat_val]
    
    # Convertir color hex a RGB
    color_hex = info['color'].lstrip('#')
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    color_rgb = (r, g, b)
    
    label = info['label']
    
    if any(not np.isnan(v) for v in valores_ndvi):
        ax.plot(meses, valores_ndvi, marker='o', label=label, 
               color=color_rgb, linewidth=2, markersize=6)

ax.set_xticks(range(len(meses)))
ax.set_xticklabels(meses, rotation=45, ha='right')
ax.set_xlabel("Mes", fontsize=12)
ax.set_ylabel("NDVI promedio", fontsize=12)
ax.set_title("Evolución temporal NDVI por categoría - VERANO 2024 - Coronel Suárez", 
            fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=9, loc='best', ncol=1)
ax.set_ylim(bottom=0)

plt.suptitle("Evolución temporal del NDVI por todas las categorías del INTA presentes en Coronel Suárez", 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Guardar figura
output_path = OUTPUT_DIR / "evolucion_temporal_ndvi_categorias_coronel_suarez.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Visualización guardada en: {output_path}")

plt.show()

print(f"\nGraficos generados exitosamente")

