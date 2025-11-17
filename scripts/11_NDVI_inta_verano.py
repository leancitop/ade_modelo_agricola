"""
Script para combinar el raster INTA de verano con los rasters NDVI desde diciembre 2023
en adelante, recalculando las estadísticas (mediana, min, max, sd) solo con esos rasters.

Maneja el problema de nodata en las primeras columnas de los rasters de enero y febrero 2024,
ajustando el recorte para que todos los rasters queden completos.
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import gc

# Paths relativos desde el script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "proc")
INTA_DIR = os.path.join(RAW_DIR, "INTA_23_24")
NDVI_DIR = os.path.join(RAW_DIR, "sentinel_23_24")

# Archivos de entrada
RECORTE_VERANO_PATH = os.path.join(PROC_DIR, "recorte_verano_GTiff.tif")

# Archivo de salida
OUTPUT_PATH = os.path.join(PROC_DIR, "11_NDVI_inta_verano.tif")

# Rasters NDVI desde diciembre 2023 en adelante
NDVI_MESES = [
    "2023-12",
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06"
]

print("=" * 80)
print("COMBINACION INTA VERANO CON NDVI DESDE DICIEMBRE 2023")
print("=" * 80)

# ============================================================================
# PASO 1: Verificar archivos y obtener rasters NDVI
# ============================================================================
print("\n" + "=" * 80)
print("PASO 1: VERIFICANDO ARCHIVOS")
print("=" * 80)

# Verificar recorte verano
if not os.path.exists(RECORTE_VERANO_PATH):
    print(f"\n[ERROR] No se encuentra el archivo: {RECORTE_VERANO_PATH}")
    exit(1)

# Obtener paths de rasters NDVI
ndvi_files = []
ndvi_nombres = []

for mes in NDVI_MESES:
    archivo = os.path.join(NDVI_DIR, f"NDVI_{mes}.tif")
    if os.path.exists(archivo):
        ndvi_files.append(archivo)
        ndvi_nombres.append(f"NDVI_{mes}")
    else:
        print(f"\n[WARN] No se encuentra el archivo: {archivo}")

if not ndvi_files:
    print("\n[ERROR] No se encontraron rasters NDVI válidos.")
    exit(1)

print(f"\nRasters NDVI encontrados: {len(ndvi_files)}")
for archivo, nombre in zip(ndvi_files, ndvi_nombres):
    print(f"  {nombre}: {os.path.basename(archivo)}")

# ============================================================================
# PASO 2: Leer parámetros de referencia y detectar recorte común
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DETECTANDO RECORTE COMUN")
print("=" * 80)

# Leer parámetros del primer raster NDVI como referencia espacial
print("\nObteniendo parametros de referencia del primer raster NDVI...")
with rasterio.open(ndvi_files[0]) as src_ndvi_ref:
    ref_meta = src_ndvi_ref.meta.copy()
    ref_crs = src_ndvi_ref.crs
    ref_transform = src_ndvi_ref.transform
    ref_width = src_ndvi_ref.width
    ref_height = src_ndvi_ref.height
    ref_bounds = src_ndvi_ref.bounds
    ref_nodata = src_ndvi_ref.nodata

print(f"\nParametros de referencia (primer raster NDVI):")
print(f"  Archivo: {os.path.basename(ndvi_files[0])}")
print(f"  Dimensiones: {ref_width} x {ref_height}")
print(f"  CRS: {ref_crs}")
print(f"  Bounds: {ref_bounds}")

# Verificar que el recorte verano tenga el mismo CRS
with rasterio.open(RECORTE_VERANO_PATH) as src_verano:
    verano_crs = src_verano.crs
    verano_width = src_verano.width
    verano_height = src_verano.height
    verano_nodata = src_verano.nodata

if verano_crs != ref_crs:
    print(f"\n[WARN] CRS diferente entre recorte verano ({verano_crs}) y NDVI ({ref_crs})")
    print("  Se asume que tienen la misma proyección espacial.")

# Leer todos los rasters NDVI para detectar áreas con nodata
print("\nAnalizando rasters NDVI para detectar áreas con nodata...")
areas_validas = []

for archivo in tqdm(ndvi_files, desc="Analizando rasters"):
    with rasterio.open(archivo) as src:
        data = src.read(1).astype(np.float32)
        
        # Manejar nodata
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        
        # Detectar filas y columnas válidas (no todas son nan)
        filas_validas = ~np.all(np.isnan(data), axis=1)
        cols_validas = ~np.all(np.isnan(data), axis=0)
        
        # Encontrar índices de inicio y fin de área válida
        if np.any(filas_validas) and np.any(cols_validas):
            fila_inicio = np.argmax(filas_validas)
            fila_fin = len(filas_validas) - np.argmax(filas_validas[::-1])
            col_inicio = np.argmax(cols_validas)
            col_fin = len(cols_validas) - np.argmax(cols_validas[::-1])
            
            areas_validas.append({
                'fila_inicio': fila_inicio,
                'fila_fin': fila_fin,
                'col_inicio': col_inicio,
                'col_fin': col_fin,
                'width': col_fin - col_inicio,
                'height': fila_fin - fila_inicio
            })
        else:
            print(f"\n[WARN] Raster {os.path.basename(archivo)} no tiene datos válidos")
            areas_validas.append(None)

# Encontrar recorte común (intersección de todas las áreas válidas)
areas_validas_filtradas = [a for a in areas_validas if a is not None]

if not areas_validas_filtradas:
    print("\n[ERROR] No se encontraron áreas válidas en los rasters NDVI.")
    exit(1)

# Calcular recorte común (máximo de inicio, mínimo de fin)
recorte_comun = {
    'fila_inicio': max([a['fila_inicio'] for a in areas_validas_filtradas]),
    'fila_fin': min([a['fila_fin'] for a in areas_validas_filtradas]),
    'col_inicio': max([a['col_inicio'] for a in areas_validas_filtradas]),
    'col_fin': min([a['col_fin'] for a in areas_validas_filtradas])
}

recorte_comun['width'] = recorte_comun['col_fin'] - recorte_comun['col_inicio']
recorte_comun['height'] = recorte_comun['fila_fin'] - recorte_comun['fila_inicio']

print(f"\nRecorte común detectado:")
print(f"  Columnas: {recorte_comun['col_inicio']} a {recorte_comun['col_fin']} (width: {recorte_comun['width']})")
print(f"  Filas: {recorte_comun['fila_inicio']} a {recorte_comun['fila_fin']} (height: {recorte_comun['height']})")

# Ajustar transform para el recorte común
# Calcular nuevo transform basado en el recorte del primer raster NDVI
from rasterio.transform import xy, from_bounds

# Obtener coordenadas de las esquinas del recorte común usando el transform del primer raster NDVI
x_esquina1, y_esquina1 = xy(ref_transform, recorte_comun['fila_inicio'], recorte_comun['col_inicio'])
x_esquina2, y_esquina2 = xy(ref_transform, recorte_comun['fila_inicio'], recorte_comun['col_fin'])
x_esquina3, y_esquina3 = xy(ref_transform, recorte_comun['fila_fin'], recorte_comun['col_inicio'])
x_esquina4, y_esquina4 = xy(ref_transform, recorte_comun['fila_fin'], recorte_comun['col_fin'])

# Calcular bounds del recorte común
nuevo_bounds = (
    min(x_esquina1, x_esquina2, x_esquina3, x_esquina4),
    min(y_esquina1, y_esquina2, y_esquina3, y_esquina4),
    max(x_esquina1, x_esquina2, x_esquina3, x_esquina4),
    max(y_esquina1, y_esquina2, y_esquina3, y_esquina4)
)

# Crear nuevo transform para el recorte
nuevo_transform = from_bounds(
    nuevo_bounds[0], nuevo_bounds[1], nuevo_bounds[2], nuevo_bounds[3],
    recorte_comun['width'], recorte_comun['height']
)

print(f"\nNuevo bounds: {nuevo_bounds}")
print(f"  Dimensiones finales: {recorte_comun['width']} x {recorte_comun['height']}")

# ============================================================================
# PASO 3: Leer y recortar datos
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: LEYENDO Y RECORTANDO DATOS")
print("=" * 80)

# Crear ventana de recorte
ventana = Window(
    col_off=recorte_comun['col_inicio'],
    row_off=recorte_comun['fila_inicio'],
    width=recorte_comun['width'],
    height=recorte_comun['height']
)

# Leer recorte verano (necesita reproyección si tiene diferente CRS o transform)
print("\nLeyendo recorte_verano_GTiff.tif...")
with rasterio.open(RECORTE_VERANO_PATH) as src_verano:
    # Verificar si necesita reproyección
    if (src_verano.crs != ref_crs or 
        src_verano.width != ref_width or 
        src_verano.height != ref_height):
        print("  [INFO] Reproyectando recorte verano para coincidir con NDVI...")
        from rasterio.warp import reproject, Resampling
        
        # Leer todo el raster verano
        verano_data = src_verano.read(1).astype(np.float32)
        if verano_nodata is not None:
            verano_data[verano_data == verano_nodata] = np.nan
        
        # Crear array destino
        banda_verano = np.empty((recorte_comun['height'], recorte_comun['width']), dtype=np.float32)
        banda_verano.fill(np.nan)
        
        # Reproyectar usando la ventana común
        reproject(
            source=verano_data,
            destination=banda_verano,
            src_transform=src_verano.transform,
            src_crs=src_verano.crs,
            dst_transform=nuevo_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )
    else:
        # Mismo CRS y dimensiones, solo aplicar ventana
        banda_verano = src_verano.read(1, window=ventana).astype(np.float32)
        if verano_nodata is not None:
            banda_verano[banda_verano == verano_nodata] = np.nan
    
    print(f"  Shape: {banda_verano.shape}")
    print(f"  Valores validos: {np.sum(~np.isnan(banda_verano)):,} de {banda_verano.size:,}")

# Leer rasters NDVI
print("\nLeyendo rasters NDVI...")
bandas_ndvi = []

for archivo, nombre in tqdm(zip(ndvi_files, ndvi_nombres), 
                           total=len(ndvi_files),
                           desc="Leyendo NDVI"):
    with rasterio.open(archivo) as src:
        banda = src.read(1, window=ventana).astype(np.float32)
        
        # Manejar nodata
        if src.nodata is not None:
            banda[banda == src.nodata] = np.nan
        
        bandas_ndvi.append(banda)

print(f"  Total de bandas NDVI leidas: {len(bandas_ndvi)}")

# ============================================================================
# PASO 4: Calcular estadísticas
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: CALCULANDO ESTADISTICOS")
print("=" * 80)

print("Apilando datos NDVI...")
stack_ndvi = np.stack(bandas_ndvi, axis=0)  # Shape: (n_bandas, height, width)
print(f"  Shape del stack: {stack_ndvi.shape}")

# Calcular estadísticos solo con los rasters NDVI seleccionados
print("\nCalculando estadísticos...")
print("  Mediana...")
mediana = np.nanmedian(stack_ndvi, axis=0)

print("  Mínimo...")
minimo = np.nanmin(stack_ndvi, axis=0)

print("  Máximo...")
maximo = np.nanmax(stack_ndvi, axis=0)

print("  Desviación estándar...")
desviacion = np.nanstd(stack_ndvi, axis=0)

# Limpiar memoria
del stack_ndvi
gc.collect()

# ============================================================================
# PASO 5: Crear raster final
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: CREANDO RASTER FINAL")
print("=" * 80)

# Orden de bandas: INTA verano, estadísticas, luego NDVI temporales
bandas_finales = [banda_verano, mediana, minimo, maximo, desviacion] + bandas_ndvi
nombres_finales = ['inta_verano', 'mediana', 'min', 'max', 'sd'] + ndvi_nombres

print(f"\nTotal de bandas en el raster final: {len(bandas_finales)}")
print(f"  Banda 1: inta_verano")
print(f"  Bandas 2-5: estadísticas (mediana, min, max, sd)")
print(f"  Bandas 6-{len(bandas_finales)}: {len(ndvi_nombres)} bandas NDVI temporales")

# Preparar metadata final
meta_final = ref_meta.copy()
meta_final.update({
    'count': len(bandas_finales),
    'width': recorte_comun['width'],
    'height': recorte_comun['height'],
    'transform': nuevo_transform,
    'dtype': 'float32',
    'nodata': np.nan,
    'compress': 'lzw'
})

# Guardar raster
print(f"\nGuardando raster en: {OUTPUT_PATH}")

with rasterio.open(OUTPUT_PATH, 'w', **meta_final) as dst:
    for i, (banda, nombre) in enumerate(tqdm(zip(bandas_finales, nombres_finales), 
                                               total=len(bandas_finales),
                                               desc="Escribiendo bandas"), 1):
        dst.write(banda, i)
        dst.set_band_description(i, nombre)

print(f"\n[OK] Raster combinado guardado exitosamente.")

# Limpiar memoria
del bandas_finales, banda_verano, mediana, minimo, maximo, desviacion, bandas_ndvi
gc.collect()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

print(f"\nArchivo de salida: {OUTPUT_PATH}")
print(f"  Dimensiones: {recorte_comun['width']} x {recorte_comun['height']}")
print(f"  CRS: {ref_crs}")
print(f"  Total de bandas: {len(nombres_finales)}")
print(f"  Banda 1: inta_verano (recorte_verano_GTiff.tif)")
print(f"  Bandas 2-5: estadísticas recalculadas (mediana, min, max, sd)")
print(f"  Bandas 6-{len(nombres_finales)}: {len(ndvi_nombres)} bandas NDVI desde diciembre 2023")
print(f"  Dtype: float32")
print(f"  Nodata: nan")

# Guardar lista de nombres de bandas
nombres_path = os.path.join(PROC_DIR, "11_nombres_bandas_ndvi_inta_verano.txt")
with open(nombres_path, 'w') as f:
    for i, nombre in enumerate(nombres_finales, 1):
        f.write(f"Banda {i}: {nombre}\n")

print(f"\n[OK] Lista de nombres de bandas guardada en: {nombres_path}")
print("\n" + "=" * 80)

