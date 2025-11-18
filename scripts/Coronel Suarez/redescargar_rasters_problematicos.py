"""
Script para re-descargar rasters problemáticos de Coronel Suárez
usando el script mejorado que crea mosaicos para asegurar cobertura completa
"""

import os
import sys
from pathlib import Path

# Agregar el directorio del script al path para importar funciones
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Importar funciones del script de descarga
from importlib import import_module
import importlib.util

# Cargar el módulo de descarga
spec = importlib.util.spec_from_file_location(
    "get_rasters", 
    os.path.join(SCRIPT_DIR, "0_get_rasters_coronel_suarez.py")
)
get_rasters_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_rasters_module)

import ee
import datetime

# Rasters problemáticos identificados
RASTERS_PROBLEMATICOS = [
    "NDVI_2023-12.tif",
    "NDVI_2024-01.tif", 
    "NDVI_2024-03.tif",
    "NDVI_2024-04.tif",
    "NDVI_2024-05.tif"
]

print("=" * 80)
print("RE-DESCARGA DE RASTERS PROBLEMATICOS - CORONEL SUAREZ")
print("=" * 80)

# Inicializar GEE
if not get_rasters_module.inicializar_gee():
    print("\nEl script no puede continuar sin un proyecto configurado.")
    sys.exit(1)

# Obtener funciones necesarias
obtener_imagen_sentinel2 = get_rasters_module.obtener_imagen_sentinel2
descargar_imagen_gee = get_rasters_module.descargar_imagen_gee
OUTPUT_DIR = get_rasters_module.OUTPUT_DIR

# Mapeo de archivos a fechas
archivo_a_fecha = {
    "NDVI_2023-12.tif": datetime.date(2023, 12, 1),
    "NDVI_2024-01.tif": datetime.date(2024, 1, 1),
    "NDVI_2024-03.tif": datetime.date(2024, 3, 1),
    "NDVI_2024-04.tif": datetime.date(2024, 4, 1),
    "NDVI_2024-05.tif": datetime.date(2024, 5, 1),
}

print(f"\nRasters a re-descargar: {len(RASTERS_PROBLEMATICOS)}")
for archivo in RASTERS_PROBLEMATICOS:
    print(f"  - {archivo}")

print("\n" + "-" * 80)

for archivo in RASTERS_PROBLEMATICOS:
    fecha = archivo_a_fecha[archivo]
    fecha_str = fecha.strftime("%Y-%m-%d")
    
    print(f"\nRe-descargando: {archivo} (fecha: {fecha_str})")
    
    # Calcular rango de fechas (todo el mes)
    fecha_inicio_ee = ee.Date(fecha_str)
    
    if fecha.month == 12:
        ultimo_dia_mes = datetime.date(fecha.year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        ultimo_dia_mes = datetime.date(fecha.year, fecha.month + 1, 1) - datetime.timedelta(days=1)
    
    fecha_fin_ee = ee.Date(str(ultimo_dia_mes)).advance(1, 'day')
    
    try:
        # Obtener imagen (ahora con mosaico si es necesario)
        imagen_ndvi = obtener_imagen_sentinel2(fecha_inicio_ee, fecha_fin_ee, calcular_ndvi_band=True)
        
        # Verificar si el archivo existe y sobrescribir automáticamente
        ruta_archivo = OUTPUT_DIR / archivo
        if ruta_archivo.exists():
            print(f"  Archivo existente encontrado: {archivo}")
            print(f"  Sobrescribiendo automáticamente (rasters problemáticos)...")
            # Eliminar el archivo anterior
            ruta_archivo.unlink()
            print(f"  Archivo anterior eliminado")
        
        # Descargar
        resultado = descargar_imagen_gee(imagen_ndvi, archivo, escala=10, usar_export=False)
        
        if resultado:
            print(f"  [OK] Re-descarga completada: {archivo}")
            
            # Verificar cobertura post-descarga
            if ruta_archivo.exists():
                import rasterio
                import numpy as np
                with rasterio.open(ruta_archivo) as src:
                    data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan
                    
                    total_pixeles = data.size
                    pixeles_validos = np.sum(~np.isnan(data))
                    porcentaje_valido = 100 * pixeles_validos / total_pixeles
                    
                    print(f"  Verificación post-descarga:")
                    print(f"    Píxeles válidos: {pixeles_validos:,} / {total_pixeles:,} ({porcentaje_valido:.2f}%)")
                    
                    if porcentaje_valido < 50:
                        print(f"  [ADVERTENCIA] Aún tiene menos del 50% de píxeles válidos")
                    elif porcentaje_valido < 95:
                        print(f"  [INFO] Cobertura mejorada pero aún menor al 95%")
                    else:
                        print(f"  [OK] Cobertura adecuada (>=95%)")
        else:
            print(f"  [ERROR] No se pudo re-descargar {archivo}")
            
    except Exception as e:
        print(f"  [ERROR] Error al re-descargar {archivo}: {e}")
        continue

print("\n" + "=" * 80)
print("Proceso de re-descarga completado")
print("=" * 80)
print("\nRecomendación: Ejecuta 'verificar_cobertura_rasters.py' para verificar los resultados")

