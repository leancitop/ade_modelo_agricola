"""
Script para recortar el MNC_verano-2024.tif al área de Coronel Suárez.
Usa los bounds y transform de los rasters NDVI de Coronel Suárez como referencia.
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from tqdm import tqdm
import gc

# Paths relativos desde el script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "proc")
INTA_DIR = os.path.join(RAW_DIR, "INTA_23_24")
NDVI_DIR = os.path.join(RAW_DIR, "sentinel_23_24_coronel_suarez")

# Archivos
MNC_VERANO_PATH = os.path.join(INTA_DIR, "MNC_verano-2024.tif")
OUTPUT_PATH = os.path.join(PROC_DIR, "MNC_verano_coronel_suarez.tif")

print("=" * 80)
print("RECORTE DE MNC VERANO PARA CORONEL SUAREZ")
print("=" * 80)

# Obtener parámetros de referencia del primer raster NDVI de Coronel Suárez
print("\nObteniendo parámetros de referencia del área de Coronel Suárez...")
ndvi_files = sorted(glob.glob(os.path.join(NDVI_DIR, "NDVI_*.tif")))

if not ndvi_files:
    print(f"[ERROR] No se encontraron rasters NDVI en {NDVI_DIR}")
    exit(1)

ref_ndvi_path = ndvi_files[0]
print(f"  Usando como referencia: {os.path.basename(ref_ndvi_path)}")

with rasterio.open(ref_ndvi_path) as src_ref:
    ref_crs = src_ref.crs
    ref_transform = src_ref.transform
    ref_width = src_ref.width
    ref_height = src_ref.height
    ref_bounds = src_ref.bounds

print(f"\nParámetros de referencia:")
print(f"  Dimensiones: {ref_width} x {ref_height}")
print(f"  CRS: {ref_crs}")
print(f"  Bounds: {ref_bounds}")

# Verificar que existe el MNC
if not os.path.exists(MNC_VERANO_PATH):
    print(f"\n[ERROR] No se encuentra el archivo: {MNC_VERANO_PATH}")
    exit(1)

print(f"\nProcesando MNC_verano-2024.tif...")
print(f"  Archivo de salida: {OUTPUT_PATH}")

# Preparar metadata del raster de salida
meta_output = {
    'driver': 'GTiff',
    'height': ref_height,
    'width': ref_width,
    'count': 1,
    'dtype': 'float32',
    'crs': ref_crs,
    'transform': ref_transform,
    'compress': 'lzw',
    'nodata': np.nan
}

# Tamaño de ventana para procesamiento
CHUNK_SIZE = 2000

# Crear array destino
mnc_recortado = np.empty((ref_height, ref_width), dtype=np.float32)
mnc_recortado.fill(np.nan)

with rasterio.open(MNC_VERANO_PATH) as src_mnc:
    mnc_crs = src_mnc.crs
    mnc_nodata = src_mnc.nodata
    
    print(f"\nArchivo MNC original:")
    print(f"  Dimensiones: {src_mnc.width} x {src_mnc.height}")
    print(f"  CRS: {mnc_crs}")
    print(f"  Bounds: {src_mnc.bounds}")
    print(f"  Nodata: {mnc_nodata}")
    
    print("\nReproyectando MNC verano por ventanas para coincidir con área de Coronel Suárez...")
    
    # Calcular número de ventanas
    n_chunks_h = (ref_height + CHUNK_SIZE - 1) // CHUNK_SIZE
    n_chunks_w = (ref_width + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = n_chunks_h * n_chunks_w
    
    print(f"  Procesando por ventanas: {n_chunks_h}x{n_chunks_w} = {total_chunks} ventanas")
    
    # Procesar por ventanas
    with tqdm(total=total_chunks, desc="Reproyectando MNC verano") as pbar:
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                # Calcular límites de la ventana destino
                row_start = i * CHUNK_SIZE
                row_end = min((i + 1) * CHUNK_SIZE, ref_height)
                col_start = j * CHUNK_SIZE
                col_end = min((j + 1) * CHUNK_SIZE, ref_width)
                
                if row_end <= row_start or col_end <= col_start:
                    pbar.update(1)
                    continue
                
                # Crear ventana destino
                dst_window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                
                # Calcular bounds de la ventana destino en coordenadas
                window_bounds = rasterio.windows.bounds(dst_window, ref_transform)
                left, bottom, right, top = window_bounds
                
                # Transformar bounds al CRS del MNC
                mnc_bounds = transform_bounds(ref_crs, mnc_crs, left, bottom, right, top)
                mnc_window = from_bounds(*mnc_bounds, src_mnc.transform)
                mnc_window = mnc_window.round_lengths().round_offsets()
                
                # Asegurar que la ventana esté dentro de los bounds (verificación manual)
                col_off = int(mnc_window.col_off)
                row_off = int(mnc_window.row_off)
                width = int(mnc_window.width)
                height = int(mnc_window.height)
                
                # Ajustar ventana para que esté dentro de los bounds
                col_off_adj = max(0, col_off)
                row_off_adj = max(0, row_off)
                width_adj = min(width, src_mnc.width - col_off_adj)
                height_adj = min(height, src_mnc.height - row_off_adj)
                
                # Verificar si hay intersección válida
                if width_adj > 0 and height_adj > 0:
                    mnc_window = Window(col_off_adj, row_off_adj, width_adj, height_adj)
                    mnc_chunk_raw = src_mnc.read(1, window=mnc_window).astype(np.float32)
                    
                    # Reproyectar al tamaño de la ventana destino
                    mnc_window_bounds = rasterio.windows.bounds(mnc_window, src_mnc.transform)
                    src_transform_mnc = transform_from_bounds(
                        mnc_window_bounds[0], mnc_window_bounds[1],
                        mnc_window_bounds[2], mnc_window_bounds[3],
                        mnc_window.width, mnc_window.height
                    )
                    dst_transform_mnc = transform_from_bounds(
                        left, bottom, right, top,
                        col_end - col_start, row_end - row_start
                    )
                    
                    mnc_chunk = np.empty((row_end - row_start, col_end - col_start), dtype=np.float32)
                    reproject(
                        source=mnc_chunk_raw,
                        destination=mnc_chunk,
                        src_transform=src_transform_mnc,
                        src_crs=mnc_crs,
                        dst_transform=dst_transform_mnc,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest
                    )
                    
                    # Manejar nodata y valores de máscara (0 y 255 según script 6)
                    EXCLUDE_VALUES = [0, 255]
                    if mnc_nodata is not None:
                        mnc_chunk[mnc_chunk == mnc_nodata] = np.nan
                    # Convertir valores de máscara a NaN
                    for exclude_val in EXCLUDE_VALUES:
                        mnc_chunk[mnc_chunk == exclude_val] = np.nan
                    
                    # Copiar a la banda final (solo donde hay valores válidos)
                    mask_validos = ~np.isnan(mnc_chunk)
                    mnc_recortado[row_start:row_end, col_start:col_end][mask_validos] = mnc_chunk[mask_validos]
                    
                    del mnc_chunk, mnc_chunk_raw
                
                pbar.update(1)
            
            # Limpiar memoria cada fila de chunks
            if (i + 1) % 5 == 0:
                gc.collect()

print(f"\nGuardando raster recortado...")
print(f"  Shape final: {mnc_recortado.shape}")
print(f"  Valores válidos: {np.sum(~np.isnan(mnc_recortado)):,} de {mnc_recortado.size:,}")
pct_validos = 100 * np.sum(~np.isnan(mnc_recortado)) / mnc_recortado.size
print(f"  Porcentaje válidos: {pct_validos:.2f}%")

# Guardar el raster recortado
with rasterio.open(OUTPUT_PATH, 'w', **meta_output) as dst:
    dst.write(mnc_recortado, 1)
    dst.descriptions = ('MNC_verano_2024',)

print(f"\n[OK] Raster recortado guardado exitosamente en:")
print(f"  {OUTPUT_PATH}")

