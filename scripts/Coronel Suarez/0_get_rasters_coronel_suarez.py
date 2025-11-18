"""
Script para descargar imágenes satelitales de Google Earth Engine para Coronel Suárez
- Imágenes NDVI mensuales desde diciembre 2023 hasta junio 2024
Ubicación: Coronel Suárez, Buenos Aires, Argentina
Misma amplitud de píxeles que Tres Arroyos (buffer de 14 km)
"""

import ee
import os
import datetime
from pathlib import Path
import requests
from tqdm import tqdm
import time

# Configuración
# Ruta relativa a la raíz del proyecto
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "sentinel_23_24_coronel_suarez"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Coordenadas de Coronel Suárez, Buenos Aires, Argentina
CORONEL_SUAREZ_LON = -61.93294
CORONEL_SUAREZ_LAT = -37.45859

# Inicializar Google Earth Engine
def inicializar_gee():
    """Inicializa Google Earth Engine con manejo de errores"""
    try:
        ee.Initialize()
        print("Google Earth Engine inicializado correctamente")
        return True
    except Exception as e:
        error_msg = str(e)
        if "no project found" in error_msg.lower():
            print("="*60)
            print("ERROR: No se encontró un proyecto de Google Cloud")
            print("="*60)
            print("\nGoogle Earth Engine requiere un proyecto de Google Cloud.")
            print("\nPasos para solucionarlo:")
            print("1. Ve a https://console.cloud.google.com/")
            print("2. Crea un nuevo proyecto (o usa uno existente)")
            print("3. Anota el ID del proyecto (ejemplo: 'mi-proyecto-123456')")
            print("4. Ejecuta: earthengine set_project TU_PROYECTO_ID")
            print("\nO modifica este script y agrega tu proyecto en la línea:")
            print("   ee.Initialize(project='tu-proyecto-id')")
            print("="*60)
        else:
            print(f"Error al inicializar GEE: {e}")
        return False

if not inicializar_gee():
    print("\nEl script no puede continuar sin un proyecto configurado.")
    print("Por favor, configura un proyecto y vuelve a ejecutar el script.")
    raise SystemExit(1)

# Área de interés: buffer de 14 km alrededor del punto central
# (Misma amplitud que Tres Arroyos)
buffer_metros = 14000
punto_central = ee.Geometry.Point([CORONEL_SUAREZ_LON, CORONEL_SUAREZ_LAT])
AOI_buffer = punto_central.buffer(buffer_metros)

# Obtener el bounding box del buffer y convertirlo a un rectángulo fijo
bbox = AOI_buffer.bounds()
AOI = bbox

# Proyección fija para todas las descargas (UTM Zone 21S para Argentina)
PROYECCION_FIJA = 'EPSG:32721'

def calcular_ndvi(imagen):
    """Calcula el NDVI a partir de las bandas NIR (B8) y Red (B4) de Sentinel-2"""
    ndvi = imagen.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi

def descargar_imagen_gee(imagen, nombre_archivo, escala=10, usar_export=False):
    """
    Descarga una imagen de GEE.
    
    Si usar_export=True o el archivo es >48MB, usa Export.image.toDrive()
    Si usar_export=False, intenta getDownloadURL() primero (más rápido para archivos pequeños)
    
    Asegura que la región descargada cubra completamente el AOI.
    """
    try:
        try:
            imagen.getInfo()
        except Exception as e:
            raise Exception(f"La imagen no existe o esta vacia: {e}")
        
        # Asegurar que la imagen esté recortada al AOI antes de descargar
        # Esto garantiza que se descargue exactamente el área de interés
        imagen_recortada = imagen.clip(AOI)
        
        if usar_export:
            return descargar_con_export(imagen_recortada, nombre_archivo, escala)
        
        try:
            url = imagen_recortada.getDownloadURL({
                'scale': escala,
                'crs': PROYECCION_FIJA,
                'region': AOI,
                'format': 'GEO_TIFF'
            })
            
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            
            ruta_completa = OUTPUT_DIR / nombre_archivo
            
            total_size = int(r.headers.get('content-length', 0))
            with open(ruta_completa, 'wb') as f:
                if total_size == 0:
                    f.write(r.content)
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=nombre_archivo) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            print(f"Descargado: {nombre_archivo} ({ruta_completa.stat().st_size / (1024*1024):.2f} MB)")
            return ruta_completa
            
        except Exception as e:
            error_msg = str(e)
            if "must be less than or equal to 50331648 bytes" in error_msg:
                print(f"  Archivo muy grande para getDownloadURL, usando Export...")
                return descargar_con_export(imagen_recortada, nombre_archivo, escala)
            else:
                raise
        
    except Exception as e:
        print(f"Error al descargar {nombre_archivo}: {e}")
        return None

def descargar_con_export(imagen, nombre_archivo, escala=10):
    """
    Descarga usando Export.image.toDrive() para archivos grandes.
    Asegura que la imagen esté recortada al AOI.
    """
    try:
        nombre_tarea = nombre_archivo.replace('.tif', '').replace('-', '_')[:100]
        
        # Asegurar que la imagen esté recortada al AOI
        imagen_recortada = imagen.clip(AOI)
        
        tarea = ee.batch.Export.image.toDrive(
            image=imagen_recortada,
            description=nombre_tarea,
            folder='GEE_Exports',
            fileNamePrefix=nombre_tarea,
            scale=escala,
            region=AOI,
            crs=PROYECCION_FIJA,
            fileFormat='GeoTIFF',
            maxPixels=1e9
        )
        
        tarea.start()
        print(f"Tarea iniciada: {nombre_tarea}")
        print("Esperando a que complete la exportacion (esto puede tardar varios minutos)...")
        
        estado_anterior = None
        while tarea.active():
            estado = tarea.state
            if estado != estado_anterior:
                print(f"Estado: {estado}")
                estado_anterior = estado
            time.sleep(10)
        
        estado_final = tarea.state
        if estado_final == 'COMPLETED':
            print(f"Exportacion completada: {nombre_tarea}")
            print("NOTA: El archivo se exporto a Google Drive en la carpeta 'GEE_Exports'")
            print("Debes descargarlo manualmente desde Drive o usar la API de Google Drive")
            return True
        elif estado_final == 'FAILED':
            error = tarea.status().get('error_message', 'Error desconocido')
            raise Exception(f"La tarea fallo: {error}")
        else:
            raise Exception(f"La tarea termino con estado: {estado_final}")
        
    except Exception as e:
        print(f"Error en exportacion: {e}")
        return None

def obtener_imagen_sentinel2(fecha_inicio, fecha_fin, calcular_ndvi_band=False):
    """
    Obtiene una imagen Sentinel-2 para un rango de fechas.
    Si una sola imagen no cubre completamente el AOI, crea un mosaico.
    """
    coleccion = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterDate(fecha_inicio, fecha_fin)
                 .filterBounds(AOI)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    count = coleccion.size().getInfo()
    if count == 0:
        fecha_inicio_str = fecha_inicio.format('YYYY-MM-dd').getInfo()
        fecha_fin_str = fecha_fin.format('YYYY-MM-dd').getInfo()
        raise Exception(f"No se encontraron imágenes en el rango {fecha_inicio_str} - {fecha_fin_str}")
    
    print(f"  Encontradas {count} imágenes con <30% nubes")
    
    # Si hay múltiples imágenes, crear un mosaico para asegurar cobertura completa
    # Usar las primeras 5 imágenes con menos nubes para el mosaico
    if count > 1:
        print(f"  Creando mosaico con las primeras {min(5, count)} imágenes (menos nubes)")
        # Tomar las primeras 5 imágenes (ya ordenadas por menos nubes)
        imagenes_mosaico = coleccion.limit(5)
        
        # Usar mosaic() que toma el primer valor no-nulo de las imágenes en orden
        # Como están ordenadas por menos nubes, prioriza las mejores
        imagen = imagenes_mosaico.mosaic()
        print(f"  Mosaico creado exitosamente (usa primer valor válido de imágenes ordenadas por nubes)")
    else:
        # Si solo hay una imagen, usarla directamente
        imagen = coleccion.first()
        print(f"  Usando única imagen disponible")
    
    # Obtener información de la imagen para verificar cobertura
    try:
        # Verificar que la imagen tenga datos en el AOI
        # Esto es una verificación básica - la imagen debería tener datos si pasó filterBounds
        info_imagen = imagen.select('B4').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=AOI,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        pixeles_en_aoi = info_imagen.get('B4', 0)
        if pixeles_en_aoi == 0:
            print(f"  [ADVERTENCIA] La imagen no tiene datos en el AOI")
        else:
            print(f"  Píxeles con datos en AOI: {pixeles_en_aoi:,}")
    except Exception as e:
        print(f"  [INFO] No se pudo verificar cobertura: {e}")
    
    if calcular_ndvi_band:
        ndvi = calcular_ndvi(imagen)
        return ndvi
    else:
        return imagen.select([
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60'
        ])

def main():
    print("\n" + "="*60)
    print("DESCARGA DE IMÁGENES SATELITALES - CORONEL SUAREZ")
    print("="*60 + "\n")
    
    # Descargar imágenes NDVI mensuales desde diciembre 2023 hasta junio 2024
    print("Descargando imágenes NDVI mensuales (dic 2023 - jun 2024)...")
    print("-" * 60)
    
    meses = []
    fecha_actual = datetime.date(2023, 12, 1)
    fecha_fin = datetime.date(2024, 6, 1)
    
    while fecha_actual <= fecha_fin:
        meses.append(fecha_actual)
        if fecha_actual.month == 12:
            fecha_actual = datetime.date(fecha_actual.year + 1, 1, 1)
        else:
            fecha_actual = datetime.date(fecha_actual.year, fecha_actual.month + 1, 1)
    
    print(f"Total de meses a descargar: {len(meses)}")
    
    for fecha in meses:
        fecha_str = fecha.strftime("%Y-%m-%d")
        fecha_inicio_ee = ee.Date(fecha_str)
        
        if fecha.month == 12:
            ultimo_dia_mes = datetime.date(fecha.year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            ultimo_dia_mes = datetime.date(fecha.year, fecha.month + 1, 1) - datetime.timedelta(days=1)
        
        fecha_fin_ee = ee.Date(str(ultimo_dia_mes)).advance(1, 'day')
        
        print(f"\nProcesando: {fecha_str} (buscando en todo el mes)")
        
        try:
            imagen_ndvi = obtener_imagen_sentinel2(fecha_inicio_ee, fecha_fin_ee, calcular_ndvi_band=True)
            nombre_archivo = f"NDVI_{fecha.strftime('%Y-%m')}.tif"
            
            # Verificar si el archivo ya existe
            ruta_archivo = OUTPUT_DIR / nombre_archivo
            if ruta_archivo.exists():
                print(f"  Archivo ya existe: {nombre_archivo}")
                print(f"  ¿Deseas sobrescribirlo? (S/N) - Por defecto: N")
                respuesta = input().strip().upper()
                if respuesta != 'S':
                    print(f"  Omitiendo descarga de {nombre_archivo}")
                    continue
            
            resultado = descargar_imagen_gee(imagen_ndvi, nombre_archivo, escala=10, usar_export=False)
            
            # Verificar que el archivo descargado tenga datos válidos
            if resultado and ruta_archivo.exists():
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
                    print(f"    Dimensiones: {data.shape}")
                    print(f"    Píxeles válidos: {pixeles_validos:,} / {total_pixeles:,} ({porcentaje_valido:.2f}%)")
                    
                    if porcentaje_valido < 50:
                        print(f"  [ADVERTENCIA] El raster tiene menos del 50% de píxeles válidos")
                        print(f"    Considera re-descargar con parámetros más permisivos")
        except Exception as e:
            print(f"  [ERROR] No se pudo descargar imagen para {fecha_str}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Proceso completado")
    print(f"Archivos guardados en: {OUTPUT_DIR.absolute()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

