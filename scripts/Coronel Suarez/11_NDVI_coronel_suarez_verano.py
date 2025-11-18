"""
Script para generar el raster de Coronel Suárez con las mismas bandas que 11_NDVI_inta_verano.tif.

Estructura del raster final:
- Banda 1: inta_verano (MNC_verano-2024.tif procesado y reproyectado al área de Coronel Suárez)
- Bandas 2-5: estadísticas (mediana, min, max, sd) calculadas desde diciembre 2023
- Bandas 6-12: NDVI temporales (7 meses: dic-2023 a jun-2024)

Nota: Si mayo tiene menos del 50% de datos válidos, se interpola usando abril y junio.
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "proc")
NDVI_DIR = os.path.join(RAW_DIR, "sentinel_23_24_coronel_suarez")

# Archivo de salida
OUTPUT_PATH = os.path.join(PROC_DIR, "11_NDVI_inta_verano_coronel_suarez.tif")

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
print("GENERACION DE RASTER CORONEL SUAREZ - NDVI VERANO")
print("=" * 80)

# ============================================================================
# PASO 1: Verificar archivos y obtener rasters NDVI
# ============================================================================
print("\n" + "=" * 80)
print("PASO 1: VERIFICANDO ARCHIVOS")
print("=" * 80)

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

# Leer todos los rasters NDVI para detectar áreas con nodata
print("\nAnalizando rasters NDVI para detectar áreas con nodata...")
areas_validas = []
porcentajes_validos = []

for idx, archivo in enumerate(tqdm(ndvi_files, desc="Analizando rasters")):
    with rasterio.open(archivo) as src:
        data = src.read(1).astype(np.float32)
        
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        
        # Calcular porcentaje de datos válidos
        total_pixels = data.size
        n_validos = np.sum(~np.isnan(data))
        pct_validos = 100 * n_validos / total_pixels
        porcentajes_validos.append(pct_validos)
        
        filas_validas = ~np.all(np.isnan(data), axis=1)
        cols_validas = ~np.all(np.isnan(data), axis=0)
        
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
                'height': fila_fin - fila_inicio,
                'pct_validos': pct_validos
            })
        else:
            print(f"\n[WARN] Raster {os.path.basename(archivo)} no tiene datos válidos")
            areas_validas.append(None)
            porcentajes_validos.append(0)

# Identificar mes problemático (mayo, índice 5)
indice_mayo_problema = 5
mayo_es_problematico = (indice_mayo_problema < len(porcentajes_validos) and 
                        porcentajes_validos[indice_mayo_problema] < 50)

# Encontrar recorte común
# Si mayo es problemático, excluirlo del cálculo del recorte común
if mayo_es_problematico:
    print(f"\n[INFO] Mayo tiene solo {porcentajes_validos[indice_mayo_problema]:.2f}% de datos válidos.")
    print(f"  Calculando recorte común excluyendo mayo para maximizar el área...")
    areas_validas_filtradas = [a for i, a in enumerate(areas_validas) 
                               if a is not None and i != indice_mayo_problema]
else:
    areas_validas_filtradas = [a for a in areas_validas if a is not None]

if not areas_validas_filtradas:
    print("\n[ERROR] No se encontraron áreas válidas en los rasters NDVI.")
    exit(1)

# Usar el área completa del primer raster NDVI como referencia
# Esto maximiza el área disponible (no limitar por intersección)
print(f"\n[INFO] Usando área completa del primer raster NDVI como referencia...")
print(f"  Esto maximiza el área disponible para el raster final")

# Calcular recorte con zoom del 5% en cada borde para eliminar NaN de los bordes
porcentaje_recorte = 0.05  # 5% en cada borde

# Calcular píxeles a recortar en cada borde
pixeles_recorte_h = int(ref_height * porcentaje_recorte)
pixeles_recorte_w = int(ref_width * porcentaje_recorte)

# Aplicar recorte del 5% en cada borde
recorte_comun = {
    'fila_inicio': pixeles_recorte_h,
    'fila_fin': ref_height - pixeles_recorte_h,
    'col_inicio': pixeles_recorte_w,
    'col_fin': ref_width - pixeles_recorte_w
}

recorte_comun['width'] = recorte_comun['col_fin'] - recorte_comun['col_inicio']
recorte_comun['height'] = recorte_comun['fila_fin'] - recorte_comun['fila_inicio']

print(f"\n[INFO] Aplicando recorte del {porcentaje_recorte*100:.0f}% en cada borde para eliminar NaN...")
print(f"  Píxeles a recortar: {pixeles_recorte_h} filas (arriba/abajo), {pixeles_recorte_w} columnas (izq/der)")
print(f"\nÁrea recortada (sin bordes):")
print(f"  Columnas: {recorte_comun['col_inicio']} a {recorte_comun['col_fin']} (width: {recorte_comun['width']})")
print(f"  Filas: {recorte_comun['fila_inicio']} a {recorte_comun['fila_fin']} (height: {recorte_comun['height']})")
print(f"  Reducción: {ref_width - recorte_comun['width']} columnas, {ref_height - recorte_comun['height']} filas")

# Calcular nuevo transform y bounds para el área recortada
# El transform debe ajustarse para reflejar el recorte del 5% en cada borde
from rasterio.transform import from_bounds

# Calcular bounds del área recortada
# Obtener bounds del área completa
left_full, bottom_full, right_full, top_full = ref_bounds

# Calcular tamaño de píxel
pixel_size_x = (right_full - left_full) / ref_width
pixel_size_y = (top_full - bottom_full) / ref_height

# Calcular nuevos bounds considerando el recorte
left_new = left_full + (pixeles_recorte_w * pixel_size_x)
right_new = right_full - (pixeles_recorte_w * pixel_size_x)
bottom_new = bottom_full + (pixeles_recorte_h * pixel_size_y)
top_new = top_full - (pixeles_recorte_h * pixel_size_y)

# Crear nuevo transform para el área recortada
nuevo_transform = from_bounds(left_new, bottom_new, right_new, top_new, 
                               recorte_comun['width'], recorte_comun['height'])
nuevo_bounds = (left_new, bottom_new, right_new, top_new)

print(f"\nTransform y bounds ajustados para área recortada:")
print(f"  Bounds originales: {ref_bounds}")
print(f"  Bounds recortados: {nuevo_bounds}")
print(f"  Dimensiones finales: {recorte_comun['width']} x {recorte_comun['height']}")

# ============================================================================
# PASO 3: Leer y recortar datos
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: LEYENDO Y RECORTANDO DATOS")
print("=" * 80)

# No usar ventana de recorte - leer área completa
ventana = None

# Leer y procesar MNC_verano-2024.tif para el área de Coronel Suárez (por ventanas para evitar problemas de memoria)
MNC_VERANO_PATH = os.path.join(RAW_DIR, "INTA_23_24", "MNC_verano-2024.tif")

print("\nLeyendo y procesando MNC_verano-2024.tif para área de Coronel Suárez...")
if not os.path.exists(MNC_VERANO_PATH):
    print(f"\n[ERROR] No se encuentra el archivo: {MNC_VERANO_PATH}")
    print(f"  Usando banda dummy (todos nodata)...")
    banda_verano = np.full((recorte_comun['height'], recorte_comun['width']), np.nan, dtype=np.float32)
else:
    # Crear array destino
    banda_verano = np.empty((recorte_comun['height'], recorte_comun['width']), dtype=np.float32)
    banda_verano.fill(np.nan)
    
    # Tamaño de ventana para procesamiento
    CHUNK_SIZE = 2000
    
    with rasterio.open(MNC_VERANO_PATH) as src_verano:
        verano_crs = src_verano.crs
        verano_nodata = src_verano.nodata
        verano_bounds = src_verano.bounds
        
        print(f"  Archivo original:")
        print(f"    Dimensiones: {src_verano.width} x {src_verano.height}")
        print(f"    CRS: {verano_crs}")
        print(f"    Bounds: {verano_bounds}")
        
        print("  [INFO] Reproyectando MNC verano por ventanas para coincidir con área de Coronel Suárez...")
        from rasterio.warp import reproject, Resampling, transform_bounds
        from rasterio.windows import Window, from_bounds
        from rasterio.transform import from_bounds as transform_from_bounds
        
        # Calcular número de ventanas
        n_chunks_h = (recorte_comun['height'] + CHUNK_SIZE - 1) // CHUNK_SIZE
        n_chunks_w = (recorte_comun['width'] + CHUNK_SIZE - 1) // CHUNK_SIZE
        total_chunks = n_chunks_h * n_chunks_w
        
        print(f"  Procesando por ventanas: {n_chunks_h}x{n_chunks_w} = {total_chunks} ventanas")
        
        # Procesar por ventanas
        with tqdm(total=total_chunks, desc="Reproyectando MNC verano") as pbar:
            for i in range(n_chunks_h):
                for j in range(n_chunks_w):
                    # Calcular límites de la ventana destino
                    row_start = i * CHUNK_SIZE
                    row_end = min((i + 1) * CHUNK_SIZE, recorte_comun['height'])
                    col_start = j * CHUNK_SIZE
                    col_end = min((j + 1) * CHUNK_SIZE, recorte_comun['width'])
                    
                    if row_end <= row_start or col_end <= col_start:
                        pbar.update(1)
                        continue
                    
                    # Crear ventana destino
                    dst_window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    # Calcular bounds de la ventana destino en coordenadas
                    # Usar el transform recortado (nuevo_transform) que ya refleja el recorte del 5%
                    window_bounds = rasterio.windows.bounds(dst_window, nuevo_transform)
                    left, bottom, right, top = window_bounds
                    
                    # Transformar bounds al CRS del MNC
                    ver_bounds = transform_bounds(ref_crs, src_verano.crs, left, bottom, right, top)
                    ver_window = from_bounds(*ver_bounds, src_verano.transform)
                    ver_window = ver_window.round_lengths().round_offsets()
                    
                    # Asegurar que la ventana esté dentro de los bounds (verificación manual)
                    col_off = int(ver_window.col_off)
                    row_off = int(ver_window.row_off)
                    width = int(ver_window.width)
                    height = int(ver_window.height)
                    
                    # Ajustar ventana para que esté dentro de los bounds
                    col_off_adj = max(0, col_off)
                    row_off_adj = max(0, row_off)
                    width_adj = min(width, src_verano.width - col_off_adj)
                    height_adj = min(height, src_verano.height - row_off_adj)
                    
                    # Verificar si hay intersección válida
                    if width_adj > 0 and height_adj > 0:
                        ver_window = Window(col_off_adj, row_off_adj, width_adj, height_adj)
                        ver_chunk_raw = src_verano.read(1, window=ver_window).astype(np.float32)
                        
                        # Reproyectar al tamaño de la ventana destino
                        ver_window_bounds = rasterio.windows.bounds(ver_window, src_verano.transform)
                        src_transform_ver = transform_from_bounds(
                            ver_window_bounds[0], ver_window_bounds[1],
                            ver_window_bounds[2], ver_window_bounds[3],
                            ver_window.width, ver_window.height
                        )
                        dst_transform_ver = transform_from_bounds(
                            left, bottom, right, top,
                            col_end - col_start, row_end - row_start
                        )
                        
                        ver_chunk = np.empty((row_end - row_start, col_end - col_start), dtype=np.float32)
                        reproject(
                            source=ver_chunk_raw,
                            destination=ver_chunk,
                            src_transform=src_transform_ver,
                            src_crs=src_verano.crs,
                            dst_transform=dst_transform_ver,
                            dst_crs=ref_crs,
                            resampling=Resampling.nearest
                        )
                        
                        # Manejar nodata
                        if verano_nodata is not None:
                            ver_chunk[ver_chunk == verano_nodata] = np.nan
                        
                        # Copiar a la banda final
                        banda_verano[row_start:row_end, col_start:col_end] = ver_chunk
                        
                        del ver_chunk, ver_chunk_raw
                    
                    pbar.update(1)
                
                # Limpiar memoria cada fila de chunks
                if (i + 1) % 5 == 0:
                    gc.collect()
    
    print(f"  Shape final: {banda_verano.shape}")
    print(f"  Valores validos: {np.sum(~np.isnan(banda_verano)):,} de {banda_verano.size:,}")
    pct_validos = 100 * np.sum(~np.isnan(banda_verano)) / banda_verano.size
    print(f"  Porcentaje válidos: {pct_validos:.2f}%")

# Leer rasters NDVI y aplicar recorte del 5% en cada borde
print("\nLeyendo rasters NDVI y aplicando recorte del 5% en cada borde...")
bandas_ndvi = []

for archivo, nombre in tqdm(zip(ndvi_files, ndvi_nombres), 
                           total=len(ndvi_files),
                           desc="Leyendo y recortando NDVI"):
    with rasterio.open(archivo) as src:
        # Leer área completa primero
        banda_completa = src.read(1).astype(np.float32)
        
        if src.nodata is not None:
            banda_completa[banda_completa == src.nodata] = np.nan
        
        # Aplicar recorte del 5% en cada borde
        banda_recortada = banda_completa[
            recorte_comun['fila_inicio']:recorte_comun['fila_fin'],
            recorte_comun['col_inicio']:recorte_comun['col_fin']
        ]
        
        bandas_ndvi.append(banda_recortada)
        
        del banda_completa, banda_recortada

print(f"  Total de bandas NDVI leidas y recortadas: {len(bandas_ndvi)}")
print(f"  Dimensiones después del recorte: {bandas_ndvi[0].shape}")

# ============================================================================
# PASO 3.5: Interpolación temporal de mayo si es necesario
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3.5: VERIFICANDO E INTERPOLANDO MAYO 2024")
print("=" * 80)

# Índices en la lista: dic=0, ene=1, feb=2, mar=3, abr=4, may=5, jun=6
indice_mayo = 5  # NDVI_2024-05 es el 6to elemento (índice 5)
indice_abril = 4  # NDVI_2024-04 es el 5to elemento (índice 4)
indice_junio = 6  # NDVI_2024-06 es el 7mo elemento (índice 6)

if len(bandas_ndvi) > indice_mayo:
    banda_mayo = bandas_ndvi[indice_mayo].copy()
    total_pixels = banda_mayo.size
    n_validos_mayo = np.sum(~np.isnan(banda_mayo))
    pct_validos_mayo = 100 * n_validos_mayo / total_pixels
    
    print(f"\nAnálisis de mayo 2024:")
    print(f"  Píxeles válidos: {n_validos_mayo:,} de {total_pixels:,} ({pct_validos_mayo:.2f}%)")
    
    # Si mayo tiene menos del 50% de datos válidos, interpolar
    if pct_validos_mayo < 50:
        print(f"\n[INFO] Mayo tiene menos del 50% de datos válidos. Aplicando interpolación temporal...")
        
        # Obtener abril y junio
        if len(bandas_ndvi) > indice_junio and len(bandas_ndvi) > indice_abril:
            banda_abril = bandas_ndvi[indice_abril].copy()
            banda_junio = bandas_ndvi[indice_junio].copy()
            
            # Interpolación temporal lineal: mayo = (abril + junio) / 2
            # Mantener datos originales de mayo donde existan
            print(f"  Interpolando usando abril y junio...")
            
            # Crear máscara de píxeles que necesitan interpolación (NaN en mayo)
            mask_necesita_interp = np.isnan(banda_mayo)
            n_necesita_interp = np.sum(mask_necesita_interp)
            
            # Interpolación: promedio de abril y junio
            # Si uno de los dos es NaN, usar el otro
            # Si ambos son NaN, dejar NaN
            mayo_interpolado = banda_mayo.copy()
            
            # Caso 1: Ambos abril y junio tienen datos válidos
            mask_ambos_validos = mask_necesita_interp & ~np.isnan(banda_abril) & ~np.isnan(banda_junio)
            mayo_interpolado[mask_ambos_validos] = (banda_abril[mask_ambos_validos] + banda_junio[mask_ambos_validos]) / 2.0
            
            # Caso 2: Solo abril tiene datos
            mask_solo_abril = mask_necesita_interp & ~np.isnan(banda_abril) & np.isnan(banda_junio)
            mayo_interpolado[mask_solo_abril] = banda_abril[mask_solo_abril]
            
            # Caso 3: Solo junio tiene datos
            mask_solo_junio = mask_necesita_interp & np.isnan(banda_abril) & ~np.isnan(banda_junio)
            mayo_interpolado[mask_solo_junio] = banda_junio[mask_solo_junio]
            
            # Reemplazar la banda de mayo con la interpolada
            bandas_ndvi[indice_mayo] = mayo_interpolado
            
            # Estadísticas de interpolación
            n_interpolados = np.sum(mask_ambos_validos | mask_solo_abril | mask_solo_junio)
            n_originales = n_validos_mayo
            n_final_validos = np.sum(~np.isnan(mayo_interpolado))
            
            print(f"\n  Resultados de interpolación:")
            print(f"    Píxeles originales mantenidos: {n_originales:,}")
            print(f"    Píxeles interpolados: {n_interpolados:,}")
            print(f"    Píxeles válidos finales: {n_final_validos:,} de {total_pixels:,} ({100*n_final_validos/total_pixels:.2f}%)")
            
            if n_final_validos < total_pixels:
                n_restantes_nan = total_pixels - n_final_validos
                print(f"    Píxeles que permanecen NaN: {n_restantes_nan:,} (abril y junio también NaN)")
        else:
            print(f"\n[WARN] No se pueden obtener abril o junio para interpolación")
    else:
        print(f"\n[INFO] Mayo tiene suficiente cobertura ({pct_validos_mayo:.2f}%). No se requiere interpolación.")
else:
    print(f"\n[WARN] No se encontró la banda de mayo en la lista")

# ============================================================================
# PASO 4: Calcular estadísticas
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: CALCULANDO ESTADISTICOS")
print("=" * 80)

print("Apilando datos NDVI...")
stack_ndvi = np.stack(bandas_ndvi, axis=0)
print(f"  Shape del stack: {stack_ndvi.shape}")

print("\nCalculando estadísticos...")
print("  Mediana...")
mediana = np.nanmedian(stack_ndvi, axis=0)

print("  Mínimo...")
minimo = np.nanmin(stack_ndvi, axis=0)

print("  Máximo...")
maximo = np.nanmax(stack_ndvi, axis=0)

print("  Desviación estándar...")
desviacion = np.nanstd(stack_ndvi, axis=0)

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
print(f"  Banda 1: inta_verano (recorte_verano_GTiff.tif reproyectado)")
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
print(f"  Banda 1: inta_verano (MNC_verano-2024.tif procesado y reproyectado al área de Coronel Suárez)")
print(f"  Bandas 2-5: estadísticas recalculadas (mediana, min, max, sd)")
print(f"  Bandas 6-{len(nombres_finales)}: {len(ndvi_nombres)} bandas NDVI desde diciembre 2023")
print(f"  Dtype: float32")
print(f"  Nodata: nan")

# Guardar lista de nombres de bandas
nombres_path = os.path.join(PROC_DIR, "11_nombres_bandas_ndvi_coronel_suarez.txt")
with open(nombres_path, 'w') as f:
    for i, nombre in enumerate(nombres_finales, 1):
        f.write(f"Banda {i}: {nombre}\n")

print(f"\n[OK] Lista de nombres de bandas guardada en: {nombres_path}")
print("\n" + "=" * 80)

