"""
Script para generar histogramas de estadísticas NDVI segregados por categorías del INTA
para el área de Coronel Suárez (similar al EDA inicial)
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
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
print("HISTOGRAMAS DE ESTADISTICAS NDVI - CORONEL SUAREZ")
print("=" * 80)

# Nombres de las bandas estadísticas (bandas 2-5)
nombres_bandas = {
    2: "Mediana",
    3: "Mínimo",
    4: "Máximo",
    5: "Desviación estándar"
}

# Valores a excluir
valores_excluir = [255, 0, 25, 31]

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

# Leer el raster
print(f"\nLeyendo raster: {RASTER_PATH}")
with rasterio.open(RASTER_PATH) as src:
    # Leer banda 1 (inta verano)
    banda_verano = src.read(1)
    
    # Leer bandas 2, 3, 4, 5 (estadísticas)
    bandas_estadisticas = {}
    for i in range(2, 6):
        banda = src.read(i)
        # Manejar nodata
        if src.nodata is not None:
            banda = banda.astype(np.float32)
            banda[banda == src.nodata] = np.nan
        else:
            banda = banda.astype(np.float32)
        bandas_estadisticas[i] = banda
    
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

# Crear visualización 1 fila x 4 columnas (solo verano)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Crear histogramas para VERANO
for col_idx, (banda_num, nombre) in enumerate(nombres_bandas.items()):
    ax = axes[col_idx]
    datos_estadistica = bandas_estadisticas[banda_num]
    
    # Crear histograma para cada categoría de verano
    for cat_val in sorted(categorias_presentes_ver.keys()):
        info = categorias_presentes_ver[cat_val]
        mask_categoria = (banda_verano == cat_val)
        mask_valido = ~np.isnan(datos_estadistica)
        mask_final = mask_categoria & mask_valido
        
        if np.sum(mask_final) > 0:
            datos_categoria = datos_estadistica[mask_final]
            label_con_n = f"{info['label']} (N={info['n_pixeles']:,})"
            
            # Convertir color hex a RGB
            color_hex = info['color'].lstrip('#')
            r = int(color_hex[0:2], 16) / 255.0
            g = int(color_hex[2:4], 16) / 255.0
            b = int(color_hex[4:6], 16) / 255.0
            color_rgb = (r, g, b)
            
            ax.hist(datos_categoria, bins=50, alpha=0.6, edgecolor='black', 
                   label=label_con_n, color=color_rgb)
    
    ax.set_xlabel("Valor", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title(f"VERANO - {nombre}", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    if col_idx == 0:
        ax.legend(fontsize=8, loc='best')

plt.suptitle("Histogramas de estadísticas NDVI segregados por categorías del INTA - Coronel Suárez\nDesde diciembre 2023", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Guardar figura
output_path = OUTPUT_DIR / "histogramas_estadisticas_ndvi_coronel_suarez.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Visualización guardada en: {output_path}")

plt.show()

print(f"\nGraficos generados exitosamente usando raster 11_NDVI_inta_verano_coronel_suarez.tif")

