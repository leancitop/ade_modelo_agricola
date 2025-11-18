"""
Script para visualizar el recorte del MNC verano de Coronel Suárez usando los colores del QML
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
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

MNC_RECORTADO_PATH = os.path.join(PROC_DIR, "MNC_verano_coronel_suarez.tif")
QML_PATH = os.path.join(INTA_DIR, "MNC_ver24.qml")

print("=" * 80)
print("VISUALIZACION MNC VERANO - CORONEL SUAREZ")
print("=" * 80)

# Leer el QML y extraer los colores
print(f"\nLeyendo archivo QML: {QML_PATH}")
tree = ET.parse(QML_PATH)
root = tree.getroot()

# Buscar los items de color en el colorrampshader
color_map = {}
legend_labels = {}

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
        alpha = int(item.get('alpha', '255'))
        
        # Convertir hex a RGB
        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        
        # Normalizar a 0-1
        color_rgb = (r/255.0, g/255.0, b/255.0, alpha/255.0)
        
        color_map[value] = color_rgb
        legend_labels[value] = label
        
        print(f"  Valor {value}: {label} - RGB({r},{g},{b})")

print(f"\nTotal de categorías encontradas: {len(color_map)}")

# Leer el raster recortado
print(f"\nLeyendo raster: {MNC_RECORTADO_PATH}")
with rasterio.open(MNC_RECORTADO_PATH) as src:
    data = src.read(1).astype(np.float32)
    
    # Manejar nodata
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    # También convertir 255 a NaN (máscara transparente según QML)
    data[data == 255] = np.nan
    
    print(f"  Dimensiones: {data.shape}")
    print(f"  Valores válidos: {np.sum(~np.isnan(data)):,} de {data.size:,}")
    
    # Obtener valores únicos
    valores_unicos = np.unique(data[~np.isnan(data)])
    print(f"  Valores únicos encontrados: {sorted(valores_unicos.astype(int).tolist())}")

# Crear colormap personalizado
max_value = int(np.nanmax(data)) if np.sum(~np.isnan(data)) > 0 else 255
min_value = int(np.nanmin(data)) if np.sum(~np.isnan(data)) > 0 else 0

# Crear array de colores para todos los valores posibles
num_colors = max(256, max_value + 1)
colors = np.ones((num_colors, 4))  # RGBA
colors[:, 3] = 0  # Transparente por defecto

# Asignar colores según el mapa
for value in range(num_colors):
    if value in color_map:
        colors[value] = color_map[value]
    else:
        # Para valores no definidos, usar gris
        colors[value] = (0.5, 0.5, 0.5, 0.3)

# Crear colormap
cmap = ListedColormap(colors[:max_value+1])

# Crear figura
fig, ax = plt.subplots(figsize=(14, 12))

# Visualizar
im = ax.imshow(data, cmap=cmap, vmin=0, vmax=max_value, interpolation='nearest')
ax.set_title('MNC Verano 2024 - Coronel Suárez', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Crear leyenda con las categorías presentes
valores_presentes = sorted([int(v) for v in valores_unicos if v in color_map])
legend_elements = []

for value in valores_presentes:
    if value in legend_labels:
        color = color_map[value]
        label = legend_labels[value]
        legend_elements.append(Patch(facecolor=color[:3], label=f"{value}: {label}"))

# Agregar leyenda
if legend_elements:
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=10, framealpha=0.9, title='Categorías')

# Agregar información
info_text = f"Dimensiones: {data.shape[1]} x {data.shape[0]} píxeles\n"
info_text += f"Valores válidos: {np.sum(~np.isnan(data)):,} ({100*np.sum(~np.isnan(data))/data.size:.1f}%)"
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Guardar figura
output_path = OUTPUT_DIR / "MNC_verano_coronel_suarez.png"

plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Visualización guardada en: {output_path}")

plt.show()

print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)
print(f"Archivo visualizado: {MNC_RECORTADO_PATH}")
print(f"Estilo aplicado desde: {QML_PATH}")
print(f"Categorías presentes: {len(valores_presentes)}")
print(f"Imagen guardada: {output_path}")

