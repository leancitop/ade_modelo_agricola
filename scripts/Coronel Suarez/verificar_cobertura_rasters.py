"""
Script para verificar la cobertura completa de los rasters descargados
y identificar cuáles necesitan ser re-descargados
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import ee

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RASTER_DIR = os.path.join(DATA_DIR, "sentinel_23_24_coronel_suarez")

# Coordenadas de Coronel Suárez
CORONEL_SUAREZ_LON = -61.93294
CORONEL_SUAREZ_LAT = -37.45859
buffer_metros = 14000

print("=" * 80)
print("VERIFICACION DE COBERTURA DE RASTERS - CORONEL SUAREZ")
print("=" * 80)

# Inicializar GEE para obtener el AOI esperado
try:
    ee.Initialize()
    punto_central = ee.Geometry.Point([CORONEL_SUAREZ_LON, CORONEL_SUAREZ_LAT])
    AOI_buffer = punto_central.buffer(buffer_metros)
    AOI = AOI_buffer.bounds()
    
    # Obtener bounds del AOI
    bounds_info = AOI.getInfo()['coordinates'][0]
    lons = [coord[0] for coord in bounds_info]
    lats = [coord[1] for coord in bounds_info]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    print(f"\nÁrea de interés esperada:")
    print(f"  Longitud: {min_lon:.6f} a {max_lon:.6f}")
    print(f"  Latitud: {min_lat:.6f} a {max_lat:.6f}")
    print(f"  Buffer: {buffer_metros} metros")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar GEE: {e}")
    print("  Continuando con verificación de archivos locales...")
    min_lon = max_lon = min_lat = max_lat = None

# Verificar archivos descargados
raster_files = sorted(Path(RASTER_DIR).glob("NDVI_*.tif"))

if not raster_files:
    print(f"\n[ERROR] No se encontraron archivos NDVI en: {RASTER_DIR}")
    exit(1)

print(f"\nEncontrados {len(raster_files)} archivos NDVI")
print("-" * 80)

rasters_problematicos = []

for archivo in raster_files:
    print(f"\nAnalizando: {archivo.name}")
    
    try:
        with rasterio.open(archivo) as src:
            # Información básica
            height, width = src.height, src.width
            bounds = src.bounds
            crs = src.crs
            
            print(f"  Dimensiones: {height} x {width}")
            print(f"  CRS: {crs}")
            print(f"  Bounds: {bounds}")
            
            # Leer datos
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            
            # Estadísticas de píxeles válidos
            total_pixeles = data.size
            pixeles_validos = np.sum(~np.isnan(data))
            pixeles_nan = np.sum(np.isnan(data))
            porcentaje_valido = 100 * pixeles_validos / total_pixeles
            
            print(f"  Píxeles válidos: {pixeles_validos:,} / {total_pixeles:,} ({porcentaje_valido:.2f}%)")
            
            # Verificar si hay áreas grandes con NaN (posible corte)
            # Detectar filas y columnas completamente vacías
            filas_vacias = np.all(np.isnan(data), axis=1)
            cols_vacias = np.all(np.isnan(data), axis=0)
            
            n_filas_vacias = np.sum(filas_vacias)
            n_cols_vacias = np.sum(cols_vacias)
            
            # Detectar áreas con muchos NaN en los bordes (indicador de corte)
            # Verificar bordes (primeras y últimas 50 filas/columnas)
            borde_size = min(50, height // 10, width // 10)
            
            if borde_size > 0:
                borde_superior = data[:borde_size, :]
                borde_inferior = data[-borde_size:, :]
                borde_izquierdo = data[:, :borde_size]
                borde_derecho = data[:, -borde_size:]
                
                nan_borde_sup = np.sum(np.isnan(borde_superior)) / borde_superior.size
                nan_borde_inf = np.sum(np.isnan(borde_inferior)) / borde_inferior.size
                nan_borde_izq = np.sum(np.isnan(borde_izquierdo)) / borde_izquierdo.size
                nan_borde_der = np.sum(np.isnan(borde_derecho)) / borde_derecho.size
                
                print(f"  Análisis de bordes (primeras/últimas {borde_size} filas/cols):")
                print(f"    Borde superior: {100*nan_borde_sup:.1f}% NaN")
                print(f"    Borde inferior: {100*nan_borde_inf:.1f}% NaN")
                print(f"    Borde izquierdo: {100*nan_borde_izq:.1f}% NaN")
                print(f"    Borde derecho: {100*nan_borde_der:.1f}% NaN")
                
                # Si hay muchos NaN en los bordes, puede ser un problema
                if nan_borde_sup > 0.5 or nan_borde_inf > 0.5 or nan_borde_izq > 0.5 or nan_borde_der > 0.5:
                    print(f"  [ADVERTENCIA] Bordes con muchos NaN - posible corte")
            
            # Verificar si hay filas/columnas completamente vacías
            if n_filas_vacias > 0:
                print(f"  [ADVERTENCIA] {n_filas_vacias} filas completamente vacías")
            if n_cols_vacias > 0:
                print(f"  [ADVERTENCIA] {n_cols_vacias} columnas completamente vacías")
            
            # Verificar cobertura geográfica
            if min_lon is not None:
                # Calcular tamaño esperado del área en metros (aproximado)
                # El buffer de 14km debería dar aproximadamente 28km x 28km
                # Con escala de 10m, debería ser aproximadamente 2800 x 2800 píxeles
                tamaño_esperado_aprox = (buffer_metros * 2 / 10)  # Escala de 10m
                
                if width < tamaño_esperado_aprox * 0.9 or height < tamaño_esperado_aprox * 0.9:
                    print(f"  [ADVERTENCIA] Dimensiones menores a las esperadas")
                    print(f"    Esperado: ~{tamaño_esperado_aprox:.0f} x {tamaño_esperado_aprox:.0f}")
                    print(f"    Obtenido: {width} x {height}")
            
            # Clasificar el raster
            if porcentaje_valido < 50:
                print(f"  [PROBLEMA] Menos del 50% de píxeles válidos")
                rasters_problematicos.append((archivo.name, "Baja cobertura de datos"))
            elif n_filas_vacias > height * 0.1 or n_cols_vacias > width * 0.1:
                print(f"  [PROBLEMA] Muchas filas/columnas vacías")
                rasters_problematicos.append((archivo.name, "Áreas vacías significativas"))
            elif porcentaje_valido < 95:
                print(f"  [ADVERTENCIA] Cobertura menor al 95%")
            else:
                print(f"  [OK] Cobertura adecuada")
                
    except Exception as e:
        print(f"  [ERROR] No se pudo leer el archivo: {e}")
        rasters_problematicos.append((archivo.name, f"Error al leer: {e}"))

# Resumen
print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

if rasters_problematicos:
    print(f"\n[ATENCION] Se encontraron {len(rasters_problematicos)} rasters problemáticos:")
    for nombre, problema in rasters_problematicos:
        print(f"  - {nombre}: {problema}")
    print(f"\nRecomendación: Re-descargar estos rasters usando el script mejorado")
else:
    print("\n[OK] Todos los rasters tienen cobertura adecuada")

print(f"\nTotal de archivos analizados: {len(raster_files)}")
print(f"Archivos problemáticos: {len(rasters_problematicos)}")

