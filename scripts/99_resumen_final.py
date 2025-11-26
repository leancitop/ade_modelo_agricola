# %% [markdown]
# # Resumen General del Trabajo: Clasificación de Cultivos Agrícolas mediante Random Forest
# 
# Este notebook presenta un resumen general del desarrollo de nuestro proyecto de clasificación de cultivos agrícolas utilizando imágenes satelitales Sentinel-2 y aprendizaje automático.
# 
# ## Objetivo del Proyecto
# 
# Nuestro objetivo es desarrollar un clasificador automático que identifique y clasifique campos agrícolas a partir de imágenes multitemporales de Sentinel-2. Para esto decidimos utilizar:
# 
# - Datos de referencia del INTA (Mapa Nacional de Cultivos) como etiquetas de verdad terreno
# - Características espectrales y fenológicas derivadas del NDVI para capturar la variabilidad temporal
# - Modelos de aprendizaje supervisado (Random Forest por píxel) para la clasificación
# 
# ## Estructura del Pipeline
# 
# El proyecto sigue un pipeline secuencial que fuimos desarrollando paso a paso:
# 
# 1. **Descarga de imágenes satelitales** desde Google Earth Engine
# 2. **Integración con datos INTA** (Mapa Nacional de Cultivos) como referencia
# 3. **Procesamiento de características multitemporales** (estadísticas/series NDVI)
# 4. **Entrenamiento del modelo Random Forest** con validación espacial
# 5. **Predicciones del modelo** para mapas de clasificación
# 6. **Post-procesamiento** (Moving Window 3x3 y CEWS)
# 7. **Validación en nuevas zonas** (generalización del modelo)

# %%
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import xml.etree.ElementTree as ET
from collections import defaultdict
from scipy.ndimage import generic_filter
import warnings
warnings.filterwarnings('ignore')

# Configuracion de paths
PROJECT_ROOT = os.path.join("..")
DATA_PROC_DIR = os.path.join(PROJECT_ROOT, "data", "proc")
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

print("Configuracion de paths:")
print(f"  Directorio de procesados: {DATA_PROC_DIR}")
print(f"  Directorio de datos raw: {DATA_RAW_DIR}")

# %% [markdown]
# ## Paso 1: Descarga de Imágenes Satelitales
# 
# Comenzamos descargando imágenes satelitales de Sentinel-2 desde Google Earth Engine. Decidimos enfocarnos en Tres Arroyos, Buenos Aires, como área de estudio inicial.
# 
# ### Datos que Descargamos
# 
# - **12 imágenes NDVI mensuales**: Junio 2023 - Junio 2024 (día 1 de cada mes) para capturar la variabilidad temporal
# - **1 imagen Sentinel-2 completa**: 13 bandas del 1 de enero de 2024 para análisis espectral
# - **Área de estudio**: Tres Arroyos, Buenos Aires, Argentina (buffer de 14 km alrededor del punto central)
# - **Proyección**: fijamos las imagenes en UTM Zone 21S (EPSG:32721) para mantener una consistencia espacial
# 
# ### Script que Utilizamos
# 
# - `scripts/0_get_rasters.py`: Script principal que desarrollamos para descargar desde Google Earth Engine
# 
# ### Visualización de Imagen NDVI Mensual
# 
# A continuación mostramos un ejemplo de una de las imágenes NDVI mensuales que descargamos:

# %%
# Visualizar una imagen NDVI mensual de ejemplo
ndvi_dir = os.path.join(DATA_RAW_DIR, "sentinel_23_24")
ndvi_files = [f for f in os.listdir(ndvi_dir) if f.startswith("NDVI_") and f.endswith(".tif")]

if ndvi_files:
    # Tomar la primera imagen disponible
    ndvi_path = os.path.join(ndvi_dir, ndvi_files[0])
    
    # Ajuste de memoria: reducir tamaño para visualizacion
    W_preview, H_preview = 1000, 1000
    
    with rasterio.open(ndvi_path) as src:
        # Leer raster con resampling para evitar problemas de memoria
        ndvi_data = src.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Filtrar valores invalidos
        ndvi_data = ndvi_data.astype(np.float32)
        ndvi_data[ndvi_data < -1] = np.nan
        ndvi_data[ndvi_data > 1] = np.nan
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.imshow(ndvi_data, cmap='RdYlGn', vmin=-0.2, vmax=0.9, interpolation='nearest', 
                       extent=extent, origin='upper')
        ax.set_title(f'NDVI Mensual - {ndvi_files[0].replace("NDVI_", "").replace(".tif", "")}', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='NDVI', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
        
        print(f"Archivo: {ndvi_files[0]}")
        print(f"Dimensiones originales: {src.height} x {src.width}")
        print(f"Dimensiones visualizacion: {H_preview} x {W_preview}")
        print(f"CRS: {src.crs}")
        print(f"NDVI min: {np.nanmin(ndvi_data):.3f}, max: {np.nanmax(ndvi_data):.3f}")
else:
    print("No se encontraron archivos NDVI en el directorio")

# %% [markdown]
# ## Paso 2: Integración con Datos del INTA
# 
# Para tener datos de referencia confiables, decidimos utilizar los datos del INTA (Mapa Nacional de Cultivos) como etiquetas de verdad terreno. Estos mapas nos permiten entrenar y validar nuestro modelo con información oficial sobre los cultivos.
# 
# ### Datos INTA que Utilizamos
# 
# - **MNC Invierno 2023**: Categorías de cultivos de invierno de la campaña 2023/2024
# - **MNC Verano 2024**: Categorías de cultivos de verano de la campaña 2023/2024
# 
# ### Procesamiento que Realizamos
# 
# Los rasters del INTA vienen en una proyección diferente y cubren toda la Argentina, así que tuvimos que recortarlos y alinearlos con nuestras imágenes NDVI para asegurar consistencia espacial. Esto fue importante para que cada píxel de nuestras imágenes satelitales correspondiera correctamente con las categorías del INTA.
# 
# ### Visualización del Mapa Nacional de Cultivos (Verano e Invierno)
# 
# A continuación mostramos los rasters del INTA para las campañas de verano 2024 e invierno 2023 que utilizamos:

# %%
# Visualizar MNC Verano e Invierno lado a lado (igual que en EDA)
mnc_verano_path = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_verano-2024.tif")
mnc_invierno_path = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_invierno2023.tif")
qml_path_ver = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_ver24.qml")
qml_path_inv = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_inv23.qml")

if os.path.exists(mnc_verano_path) and os.path.exists(mnc_invierno_path):
    # --- Parse QML para VERANO ---
    tree = ET.parse(qml_path_ver)
    root = tree.getroot()
    
    # Encontrar el nodo <colorrampshader>
    shader = root.find(".//colorrampshader")
    items = shader.findall(".//item")
    
    # Extraer valores, etiquetas y colores
    labels = []
    vals = []
    hex_colors = []
    for it in items:
        vals.append(int(float(it.get("value"))))
        labels.append(it.get("label"))
        qml_col = it.get("color")
        if qml_col.startswith("#"):  # hexadecimal color
            hex_colors.append(qml_col.lower())
        else:
            # formateo "R,G,B,A"
            rgb_split = qml_col.split(",")
            if len(rgb_split) >= 3:
                try:
                    rgb = tuple(map(int, rgb_split[:3]))
                    hex_colors.append('#{:02x}{:02x}{:02x}'.format(*rgb))
                except Exception as e:
                    hex_colors.append("#888888")
                    print(f"[WARN] No se pudo interpretar color '{qml_col}': {e}")
            else:
                hex_colors.append("#888888")
                print(f"[WARN] Color desconocido para item con label '{it.get('label')}': '{qml_col}'")
    
    # Ordenar por valor
    z = sorted(zip(vals, hex_colors, labels), key=lambda t: t[0])
    vals, hex_colors, labels = [list(x) for x in zip(*z)]
    
    # Para BoundaryNorm, los boundaries deben ser valores exactos de corte de clase
    boundaries = vals + [vals[-1] + 1]
    cmap = ListedColormap(hex_colors)
    norm = BoundaryNorm(boundaries, ncolors=len(hex_colors))
    
    # --- Parse QML para INVIERNO ---
    tree_inv = ET.parse(qml_path_inv)
    root_inv = tree_inv.getroot()
    
    labels_inv = []
    vals_inv = []
    hex_colors_inv = []
    for it in root_inv.iter('item'):
        label = it.get('label')
        value = it.get('value')
        color = it.get('color')
        if label and value:
            labels_inv.append(label)
            vals_inv.append(int(float(value)))
            qml_col = color
            if qml_col is None:
                hex_colors_inv.append("#888888")
                continue
            if qml_col.startswith("#"):
                hex_colors_inv.append(qml_col)
            elif "," in qml_col:
                rgb_split = qml_col.split(",")
                try:
                    rgb = tuple(map(int, rgb_split[:3]))
                    hex_colors_inv.append('#{:02x}{:02x}{:02x}'.format(*rgb))
                except Exception as e:
                    hex_colors_inv.append("#888888")
                    print(f"[WARN] No se pudo interpretar color '{qml_col}': {e}")
            else:
                hex_colors_inv.append("#888888")
                print(f"[WARN] Color desconocido para item con label '{it.get('label')}': '{qml_col}'")
    
    # Ordenar por valor
    z_inv = sorted(zip(vals_inv, hex_colors_inv, labels_inv), key=lambda t: t[0])
    vals_inv, hex_colors_inv, labels_inv = [list(x) for x in zip(*z_inv)]
    
    # Para BoundaryNorm
    boundaries_inv = vals_inv + [vals_inv[-1] + 1]
    cmap_inv = ListedColormap(hex_colors_inv)
    norm_inv = BoundaryNorm(boundaries_inv, ncolors=len(hex_colors_inv))
    
    # --- Leer los rasters y muestrear low-res ---
    W_preview, H_preview = 1000, 1000
    
    # Leer raster VERANO
    with rasterio.open(mnc_verano_path) as src:
        img = src.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    # Leer raster INVIERNO
    with rasterio.open(mnc_invierno_path) as src_inv:
        img_inv = src_inv.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        extent_inv = [src_inv.bounds.left, src_inv.bounds.right, src_inv.bounds.bottom, src_inv.bounds.top]
    
    # --- Visualización combinada ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot izquierdo: Verano
    axes[0].imshow(img, cmap=cmap, norm=norm, extent=extent, origin='upper')
    axes[0].set_title("Mapa Categorizado: MNC_verano-2024", fontsize=15)
    axes[0].axis("off")
    handles_ver = [mpatches.Patch(color=hex_colors[i], label=labels[i]) for i in range(len(labels))]
    axes[0].legend(handles=handles_ver, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    # Plot derecho: Invierno
    axes[1].imshow(img_inv, cmap=cmap_inv, norm=norm_inv, extent=extent_inv, origin='upper')
    axes[1].set_title("Mapa Categorizado: MNC_invierno-2023", fontsize=15)
    axes[1].axis("off")
    handles_inv = [mpatches.Patch(color=hex_colors_inv[i], label=labels_inv[i]) for i in range(len(labels_inv))]
    axes[1].legend(handles=handles_inv, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Dimensiones originales Verano: {src.height} x {src.width}")
    print(f"Dimensiones originales Invierno: {src_inv.height} x {src_inv.width}")
    print(f"Dimensiones visualizacion: {H_preview} x {W_preview}")
    print(f"CRS: {src.crs}")
else:
    print("Archivos MNC no encontrados")

# %% [markdown]
# ## Paso 3: Procesamiento de Características Multitemporales
# 
# Una vez que descargamos las imágenes NDVI mensuales, necesitábamos procesarlas para generar características que capturaran tanto la variabilidad temporal como las estadísticas resumen. Esto nos permitiría entrenar un modelo que aproveche tanto los patrones fenológicos (cómo cambian los cultivos a lo largo del tiempo) como las características estadísticas que resumen el comportamiento temporal.
# 
# ### Características que Generamos
# 
# El raster final `11_NDVI_inta_verano.tif` que creamos contiene:
# 
# - **Banda 1**: Categorías INTA verano (nuestros datos de referencia para entrenamiento)
# - **Bandas 2-5**: Estadísticas NDVI calculadas sobre los meses desde diciembre 2023:
#   - Mediana (valor central)
#   - Mínimo (valor más bajo)
#   - Máximo (valor más alto)
#   - Desviación estándar (variabilidad)
# - **Bandas 6-12**: Series temporales NDVI (7 meses: dic-2023 a jun-2024) para capturar la evolución temporal
# 
# **Total: 12 bandas** (1 de referencia + 4 estadísticas + 7 temporales)
# 
# ### Scripts que Desarrollamos
# 
# - `scripts/5_combinar_rasters_ndvi.py`: Combina todos los rasters NDVI mensuales en uno solo
# - `scripts/6_recortar_mnc_y_combinar.py`: Recorta y alinea los MNC con nuestras imágenes NDVI
# - `scripts/9_combinar_recortes_con_ndvi.py`: Combina los recortes MNC con las bandas NDVI
# - `scripts/11_NDVI_inta_verano.py`: Genera el raster final con estadísticas y series temporales
# 
# ### Visualización del Raster Final de Características
# 
# A continuación mostramos algunas de las bandas del raster final que generamos:

# %%
# Visualizar raster final de caracteristicas
raster_path = os.path.join(DATA_PROC_DIR, "11_NDVI_inta_verano.tif")
qml_path_ver = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_ver24.qml")

if os.path.exists(raster_path):
    # Parsear QML para obtener nombres y colores de categorias
    tree = ET.parse(qml_path_ver)
    root = tree.getroot()
    shader = root.find(".//colorrampshader")
    items = shader.findall(".//item")
    
    # Crear diccionario de categorias: valor -> (nombre, color)
    categorias_dict = {}
    for it in items:
        valor = int(float(it.get("value")))
        nombre = it.get("label")
        qml_col = it.get("color")
        if qml_col.startswith("#"):
            color = qml_col.lower()
        elif "," in qml_col:
            rgb_split = qml_col.split(",")
            if len(rgb_split) >= 3:
                try:
                    rgb = tuple(map(int, rgb_split[:3]))
                    color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                except:
                    color = "#888888"
            else:
                color = "#888888"
        else:
            color = "#888888"
        categorias_dict[valor] = (nombre, color)
    
    # Ajuste de memoria: reducir tamaño para visualizacion
    W_preview, H_preview = 1000, 1000
    
    with rasterio.open(raster_path) as src:
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Leer diferentes bandas para visualizacion con resampling
        categorias = src.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )  # Banda 1: categorias INTA
        mediana_ndvi = src.read(
            2,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )  # Banda 2: mediana NDVI
        max_ndvi = src.read(
            4,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )  # Banda 4: maximo NDVI
        std_ndvi = src.read(
            5,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )  # Banda 5: desviacion estandar NDVI
        
        # Leer todas las bandas temporales para evolucion temporal
        bandas_temporales = []
        for i in range(6, 13):  # Bandas 6-12
            banda = src.read(
                i,
                out_shape=(H_preview, W_preview),
                resampling=rasterio.enums.Resampling.nearest
            )
            bandas_temporales.append(banda.astype(np.float32))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Categorias INTA con nombres del QML
        ax = axes[0, 0]
        categorias_display = categorias.copy().astype(np.float32)
        categorias_display[(categorias_display == 0) | (categorias_display == 255)] = np.nan
        
        # Crear colormap personalizado con colores del QML
        valores_unicos = np.unique(categorias[~np.isnan(categorias_display)])
        valores_unicos = valores_unicos[valores_unicos >= 0]
        if len(valores_unicos) > 0:
            # Crear colormap con colores del QML
            colores_lista = []
            labels_lista = []
            for val in sorted(valores_unicos):
                if val in categorias_dict:
                    nombre, color = categorias_dict[val]
                    colores_lista.append(color)
                    labels_lista.append((val, nombre))
            
            if len(colores_lista) > 0:
                cmap_custom = ListedColormap(colores_lista)
                valores_unicos_lista = sorted(valores_unicos.tolist())
                boundaries = valores_unicos_lista + [valores_unicos_lista[-1] + 1]
                norm_custom = BoundaryNorm(boundaries, ncolors=len(colores_lista))
                im1 = ax.imshow(categorias_display, cmap=cmap_custom, norm=norm_custom, 
                               interpolation='nearest', extent=extent, origin='upper')
                # Crear leyenda con nombres
                handles = [mpatches.Patch(color=colores_lista[i], label=f"{labels_lista[i][0]}: {labels_lista[i][1]}") 
                          for i in range(len(labels_lista))]
                ax.legend(handles=handles, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)
            else:
                im1 = ax.imshow(categorias_display, cmap='tab20', interpolation='nearest', 
                               extent=extent, origin='upper')
        else:
            im1 = ax.imshow(categorias_display, cmap='tab20', interpolation='nearest', 
                           extent=extent, origin='upper')
        
        ax.set_title('Banda 1: Categorias INTA Verano', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Mediana NDVI
        ax = axes[0, 1]
        mediana_ndvi_float = mediana_ndvi.astype(np.float32)
        im2 = ax.imshow(mediana_ndvi_float, cmap='RdYlGn', vmin=-0.2, vmax=0.9, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Banda 2: Mediana NDVI', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im2, ax=ax, label='NDVI', fraction=0.046, pad=0.04)
        
        # Maximo NDVI
        ax = axes[1, 0]
        max_ndvi_float = max_ndvi.astype(np.float32)
        im3 = ax.imshow(max_ndvi_float, cmap='RdYlGn', vmin=-0.2, vmax=0.9, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Banda 4: Maximo NDVI', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im3, ax=ax, label='NDVI', fraction=0.046, pad=0.04)
        
        # Desviacion Estandar NDVI
        ax = axes[1, 1]
        std_ndvi_float = std_ndvi.astype(np.float32)
        im4 = ax.imshow(std_ndvi_float, cmap='YlOrRd', vmin=0, vmax=0.3, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Banda 5: Desviacion Estandar NDVI', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im4, ax=ax, label='SD NDVI', fraction=0.046, pad=0.04)
        
        plt.suptitle('Raster Final de Caracteristicas: 11_NDVI_inta_verano.tif', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
        # Grafico de evolucion temporal general del NDVI
        meses = ["2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
        ndvi_means = []
        for banda in bandas_temporales:
            # Filtrar valores validos
            banda_valid = banda.copy()
            banda_valid[banda_valid < -1] = np.nan
            banda_valid[banda_valid > 1] = np.nan
            ndvi_mean = np.nanmean(banda_valid)
            ndvi_means.append(ndvi_mean)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(meses, ndvi_means, marker='o', color='green', linewidth=2, markersize=8)
        ax.set_xticks(range(len(meses)))
        ax.set_xticklabels(meses, rotation=45, ha='right')
        ax.set_xlabel("Mes", fontsize=12)
        ax.set_ylabel("NDVI medio", fontsize=12)
        ax.set_title("Evolucion temporal del NDVI medio (Sentinel-2)", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()
        
        print(f"Dimensiones originales: {src.height} x {src.width}")
        print(f"Dimensiones visualizacion: {H_preview} x {W_preview}")
        print(f"Numero de bandas: {src.count}")
        print(f"CRS: {src.crs}")
else:
    print("Raster de caracteristicas no encontrado")

# %% [markdown]
# ### Problemática del Barbecho y Justificación del Uso de Datos de Verano
# 
# Un hallazgo importante que descubrimos durante el análisis exploratorio es que la clasificación del INTA basada en estaciones solo es válida si se analizan los meses correspondientes a cada temporada. La clase "Barbecho" presenta un comportamiento particular: **se detecta un aumento significativo de NDVI durante el verano en los píxeles clasificados como barbecho de invierno**, y un comportamiento análogo ocurre en los píxeles clasificados como barbecho de verano durante los meses de invierno.
# 
# Este comportamiento indica que los píxeles identificados como barbecho solo cumplen con dicha categoría durante la estación asignada por INTA, mientras que en el resto del ciclo agrícola pueden corresponder a otros usos del suelo, generalmente cultivos. Esto complica el desarrollo de modelos de clasificación, ya que si se entrena un clasificador utilizando toda la serie temporal sin tener en cuenta la estacionalidad, el modelo probablemente identificará erróneamente los píxeles que cambian de uso entre estaciones.
# 
# **Por esta razón, decidimos enfocarnos únicamente en los datos de verano para entrenar nuestro modelo**, restringiendo el entrenamiento y validación a los períodos en que las etiquetas del INTA son más confiables según la campaña.
# 
# A continuación mostramos la evolución temporal del NDVI agrupado por tipo de uso del suelo para invierno y verano, que ilustra esta problemática:

# %%
# Evolucion temporal del NDVI agrupado por tipo de uso del suelo (Verano e Invierno)

raster_path_9 = os.path.join(DATA_PROC_DIR, "9_NDVI_con_recortes.tif")
qml_path_inv = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_inv23.qml")
qml_path_ver = os.path.join(DATA_RAW_DIR, "INTA_23_24", "MNC_ver24.qml")

if os.path.exists(raster_path_9):
    # Meses correspondientes a las bandas 7-19 (NDVI temporal)
    meses = [
        "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12",
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"
    ]
    
    # Funcion para parsear QML y obtener categorias
    def parsear_categorias_qml(qml_path):
        tree = ET.parse(qml_path)
        root = tree.getroot()
        categorias = {}
        for it in root.iter('item'):
            label = it.get('label')
            value = it.get('value')
            if label and value:
                cat_val = int(float(value))
                categorias[cat_val] = label
        return categorias
    
    # Parsear categorias
    categorias_inv = parsear_categorias_qml(qml_path_inv)
    categorias_ver = parsear_categorias_qml(qml_path_ver)
    
    # Definir grupos para INVIERNO
    barbecho_inv = [18]  # Barbecho
    no_agricola_inv = [20]  # No agricola
    cultivos_inv = [cat for cat in categorias_inv.keys() 
                    if cat not in barbecho_inv + no_agricola_inv + [25, 255]]
    
    # Definir grupos para VERANO
    barbecho_ver = [21]  # Barbecho
    no_agricola_ver = [22]  # No agricola
    cultivos_ver = [cat for cat in categorias_ver.keys() 
                    if cat not in barbecho_ver + no_agricola_ver + [31, 255]]
    
    # Funcion para calcular NDVI promedio por grupo
    def calcular_ndvi_por_grupo(banda_categorias, bandas_ndvi, grupos):
        resultados = defaultdict(list)
        for idx_mes, banda_ndvi in enumerate(bandas_ndvi):
            for grupo_nombre, categorias_grupo in grupos.items():
                mask_grupo = np.zeros_like(banda_categorias, dtype=bool)
                for cat_val in categorias_grupo:
                    mask_grupo |= (banda_categorias == cat_val)
                mask_valido = ~np.isnan(banda_ndvi)
                mask_final = mask_grupo & mask_valido
                if np.sum(mask_final) > 0:
                    ndvi_promedio = np.nanmean(banda_ndvi[mask_final])
                else:
                    ndvi_promedio = np.nan
                resultados[grupo_nombre].append(ndvi_promedio)
        return resultados
    
    # Leer el raster
    with rasterio.open(raster_path_9) as src:
        banda_invierno = src.read(1)
        banda_verano = src.read(2)
        bandas_ndvi = []
        for i in range(7, 20):  # Bandas 7 a 19
            banda = src.read(i)
            if src.nodata is not None:
                banda = banda.astype(np.float32)
                banda[banda == src.nodata] = np.nan
            else:
                banda = banda.astype(np.float32)
            bandas_ndvi.append(banda)
    
    # Calcular para INVIERNO
    grupos_inv = {
        'Barbecho': barbecho_inv,
        'No agricola': no_agricola_inv,
        'Cultivos': cultivos_inv
    }
    ndvi_por_grupo_inv = calcular_ndvi_por_grupo(banda_invierno, bandas_ndvi, grupos_inv)
    
    # Calcular para VERANO
    grupos_ver = {
        'Barbecho': barbecho_ver,
        'No agricola': no_agricola_ver,
        'Cultivos': cultivos_ver
    }
    ndvi_por_grupo_ver = calcular_ndvi_por_grupo(banda_verano, bandas_ndvi, grupos_ver)
    
    # Crear visualizacion
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Colores para los grupos
    colores_grupos = {
        'Barbecho': '#646b63',  # Gris
        'No agricola': '#e6f0c2',  # Beige claro
        'Cultivos': '#42f4ce'  # Verde azulado
    }
    
    # Subplot izquierdo: INVIERNO
    ax_inv = axes[0]
    for grupo_nombre in ['Barbecho', 'No agricola', 'Cultivos']:
        valores_ndvi = ndvi_por_grupo_inv[grupo_nombre]
        color = colores_grupos[grupo_nombre]
        if any(not np.isnan(v) for v in valores_ndvi):
            ax_inv.plot(meses, valores_ndvi, marker='o', label=grupo_nombre, 
                       color=color, linewidth=2.5, markersize=8)
    
    ax_inv.set_xticks(range(len(meses)))
    ax_inv.set_xticklabels(meses, rotation=45, ha='right')
    ax_inv.set_xlabel("Mes", fontsize=12)
    ax_inv.set_ylabel("NDVI promedio", fontsize=12)
    ax_inv.set_title("Evolucion temporal NDVI por grupo - INVIERNO 2023", 
                    fontsize=14, fontweight='bold')
    ax_inv.grid(True, linestyle='--', alpha=0.5)
    ax_inv.legend(fontsize=11, loc='best')
    ax_inv.set_ylim(bottom=0)
    
    # Subplot derecho: VERANO
    ax_ver = axes[1]
    for grupo_nombre in ['Barbecho', 'No agricola', 'Cultivos']:
        valores_ndvi = ndvi_por_grupo_ver[grupo_nombre]
        color = colores_grupos[grupo_nombre]
        if any(not np.isnan(v) for v in valores_ndvi):
            ax_ver.plot(meses, valores_ndvi, marker='o', label=grupo_nombre, 
                       color=color, linewidth=2.5, markersize=8)
    
    ax_ver.set_xticks(range(len(meses)))
    ax_ver.set_xticklabels(meses, rotation=45, ha='right')
    ax_ver.set_xlabel("Mes", fontsize=12)
    ax_ver.set_ylabel("NDVI promedio", fontsize=12)
    ax_ver.set_title("Evolucion temporal NDVI por grupo - VERANO 2024", 
                    fontsize=14, fontweight='bold')
    ax_ver.grid(True, linestyle='--', alpha=0.5)
    ax_ver.legend(fontsize=11, loc='best')
    ax_ver.set_ylim(bottom=0)
    
    plt.suptitle("Evolucion temporal del NDVI agrupado por tipo de uso del suelo", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("Graficos generados exitosamente")
else:
    print("Raster 9_NDVI_con_recortes.tif no encontrado")

# %% [markdown]
# ## Paso 4: Entrenamiento del Modelo Random Forest
# 
# Decidimos entrenar un modelo de Random Forest por píxel para clasificar los cultivos.
# 
# ### Agrupación de Categorías que Decidimos
# 
# Las categorías del INTA son muy específicas (Maíz, Soja, Girasol, etc.), así que decidimos agruparlas en 3 clases más generales para simplificar el problema y agrupar cultivos con características fenológicas similares:
# 
# - **0: CULTIVO AGRÍCOLA**: Maíz (10), Soja (11), Girasol (12), Sorgo (18), Papa (26)
# - **1: BARBECHO**: Barbecho (21)
# - **2: NO AGRÍCOLA**: No agrícola (22) y nodata (255)
# 
# ### Variables Predictoras que Utilizamos
# 
# - **11 features**: 4 estadísticas NDVI + 7 valores temporales NDVI (desde diciembre 2023)
# 
# ### Validación Espacial que Implementamos
# 
# Cuando trabajamos con datos espaciales, los píxeles vecinos suelen estar correlacionados. Si entrenamos con píxeles que son vecinos de los píxeles de test, el modelo puede "hacer trampa" al aprender patrones espaciales locales en lugar de patrones generalizables. Por eso implementamos una estrategia de **validación por bloques espaciales**:
# 
# - Dividimos el área en bloques 3x3 (grid)
# - Los bloques de entrenamiento y test están separados espacialmente
# - Usamos un patrón tipo tablero de ajedrez para asegurar independencia espacial
# 
# ### Hiperparámetros que Seleccionamos
# 
# - `n_estimators=50`: Lo optimizamos mediante análisis de OOB error (probamos desde 10 hasta 200)
# - `max_depth=20`: Limitamos la profundidad para evitar sobreajuste
# - `min_samples_split=5`: Evita sobreajuste en hojas pequeñas
# - `min_samples_leaf=2`: Controla la complejidad de las hojas
# - `class_weight='balanced'`: Para manejar el desbalance de clases
# 
# ### Resultados que Obtuvimos
# 
# - **Accuracy**: 86.27%
# - **Precision/Recall por clase**:
#   - CULTIVO AGRÍCOLA: 90.2% / 90.5% (la mejor clasificada)
#   - BARBECHO: 79.3% / 88.9%
#   - NO AGRÍCOLA: 81.5% / 75.6%
# 
# ### Script que Utilizamos
# 
# - `scripts/10_RF_per_pixel.ipynb`: Notebook principal donde desarrollamos y entrenamos el modelo

# %% [markdown]
# ## Paso 5: Predicciones del Modelo
# 
# Una vez que entrenamos el modelo, generamos predicciones para todos los píxeles válidos del área de estudio. Esto nos permite crear mapas completos de clasificación y evaluar visualmente qué tan bien está funcionando el modelo.
# 
# ### Visualización: Realidad vs Predicción
# 
# A continuación comparamos las categorías reales del INTA (nuestra verdad terreno) con las predicciones que genera nuestro modelo:

# %%
# Visualizar predicciones del modelo
prediccion_path = os.path.join(DATA_PROC_DIR, "11_prediccion_rf_verano.tif")
raster_caracteristicas = os.path.join(DATA_PROC_DIR, "11_NDVI_inta_verano.tif")

if os.path.exists(prediccion_path) and os.path.exists(raster_caracteristicas):
    # Ajuste de memoria: reducir tamaño para visualizacion
    W_preview, H_preview = 1000, 1000
    
    with rasterio.open(prediccion_path) as src_pred, \
         rasterio.open(raster_caracteristicas) as src_real:
        
        extent = [src_pred.bounds.left, src_pred.bounds.right, 
                 src_pred.bounds.bottom, src_pred.bounds.top]
        
        # Leer rasters con resampling para evitar problemas de memoria
        prediccion = src_pred.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        realidad = src_real.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        
        # Agrupar categorias reales para comparacion
        # 0: CULTIVO AGRICOLA, 1: BARBECHO, 2: NO AGRICOLA
        realidad_agrupada = np.full_like(realidad, -1, dtype=np.int32)
        cultivos_agricolas = [10, 11, 12, 18, 26]
        for cat in cultivos_agricolas:
            realidad_agrupada[realidad == cat] = 0
        realidad_agrupada[realidad == 21] = 1  # BARBECHO
        realidad_agrupada[(realidad == 22) | (realidad == 255)] = 2  # NO AGRICOLA
        
        # Colores para las clases
        colores_clases = {
            0: '#339820',  # Verde para CULTIVO AGRICOLA
            1: '#646b63',  # Gris para BARBECHO
            2: '#e6f0c2'   # Beige para NO AGRICOLA
        }
        cmap_custom = ListedColormap([colores_clases[0], colores_clases[1], colores_clases[2]])
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Realidad
        ax = axes[0]
        mask_valido = realidad_agrupada >= 0
        realidad_display = realidad_agrupada.copy()
        realidad_display[~mask_valido] = -1
        im1 = ax.imshow(realidad_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Realidad (INTA Verano - Agrupado)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Prediccion
        ax = axes[1]
        # Convertir a int32 para poder asignar valores negativos
        prediccion_display = prediccion.copy().astype(np.int32)
        prediccion_display[prediccion < 0] = -1
        im2 = ax.imshow(prediccion_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Prediccion (Random Forest)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Leyenda comun
        nombres_clases = ['CULTIVO AGRICOLA', 'BARBECHO', 'NO AGRICOLA']
        handles = [Patch(facecolor=colores_clases[i], label=nombres_clases[i]) for i in range(3)]
        fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.02))
        
        # Titulo general con mayor margen respecto a los titulos de cada subplot
        plt.suptitle('Comparacion: Realidad vs Prediccion - Random Forest por Pixel', 
                     fontsize=16, fontweight='bold', y=0.98)
        # Dejamos espacio en la parte superior (top=0.9) para separar el suptitle de los ejes
        plt.tight_layout(rect=[0, 0.08, 1, 0.9])
        plt.subplots_adjust(bottom=0.08)
        plt.show()
        
        # Calcular accuracy
        mask_comparacion = (realidad_agrupada >= 0) & (prediccion >= 0)
        y_true = realidad_agrupada[mask_comparacion].ravel()
        y_pred = prediccion[mask_comparacion].ravel()
        aciertos = (y_true == y_pred).sum()
        total = y_true.size
        accuracy = aciertos / total if total > 0 else 0
        
        print(f"Accuracy espacial: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Pixeles correctos: {aciertos:,} de {total:,}")

        # Matriz de confusion (filas = realidad, columnas = prediccion)
        clases = [0, 1, 2]  # 0: Cultivo agricola, 1: Barbecho, 2: No agricola
        nombres_clases = ['CULTIVO AGRICOLA', 'BARBECHO', 'NO AGRICOLA']
        n_clases = len(clases)
        matriz_confusion = np.zeros((n_clases, n_clases), dtype=np.int64)

        for i, c_real in enumerate(clases):
            for j, c_pred in enumerate(clases):
                matriz_confusion[i, j] = np.sum((y_true == c_real) & (y_pred == c_pred))

        print("\nMatriz de confusion (filas = realidad, columnas = prediccion):")
        print("          " + "  ".join([f"{nc:>16}" for nc in nombres_clases]))
        for i, nc_real in enumerate(nombres_clases):
            fila_valores = "  ".join([f"{matriz_confusion[i, j]:16d}" for j in range(n_clases)])
            print(f"{nc_real:16}  {fila_valores}")

        # Visualizacion grafica de la matriz de confusion
        fig_cm, ax_cm = plt.subplots(1, 1, figsize=(7, 6))
        im_cm = ax_cm.imshow(matriz_confusion, cmap='Blues')

        for i in range(n_clases):
            for j in range(n_clases):
                ax_cm.text(
                    j, i, f"{matriz_confusion[i, j]:,}",
                    ha='center', va='center', color='black', fontsize=10
                )

        ax_cm.set_xticks(range(n_clases))
        ax_cm.set_xticklabels(nombres_clases, rotation=45, ha='right')
        ax_cm.set_yticks(range(n_clases))
        ax_cm.set_yticklabels(nombres_clases)
        ax_cm.set_xlabel("Prediccion", fontsize=12)
        ax_cm.set_ylabel("Realidad", fontsize=12)
        ax_cm.set_title("Matriz de confusion - Tres Arroyos", fontsize=14, fontweight='bold')
        plt.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
else:
    print("Archivos de prediccion no encontrados")

# %% [markdown]
# ## Paso 6: Post-procesamiento con Moving Window y CEWS
# 
# Notamos que las predicciones del modelo tenían algunos píxeles aislados que parecían ruido (efecto "salt and pepper"). Como los campos agrícolas suelen ser áreas continuas, decidimos probar dos técnicas de post-procesamiento para suavizar las predicciones y mejorar la coherencia espacial: **Moving Window** y **CEWS**.
# 
# ### Filtro Moving Window 3x3
# 
# #### Objetivo del Filtro
# 
# - **Suavizar predicciones**: Eliminar píxeles aislados que probablemente fueron mal clasificados
# - **Mejorar coherencia espacial**: Los campos agrícolas son áreas continuas, no píxeles dispersos
# - **Reducir ruido**: Aplicar un filtro de mayoría (moda) en una ventana 3x3
# 
# #### Cómo Funciona
# 
# Para cada píxel, analizamos su vecindario 3x3 y asignamos la clase más frecuente (moda) en esa ventana. Esto hace que los píxeles aislados se "corrijan" según lo que predomina en su entorno.
# 
# #### Script que Desarrollamos
# 
# - `scripts/MWM_filter/mwm_3_arroyos.py`: Script que aplica el filtro Moving Window 3x3
# 
# ### Algoritmo CEWS (Canny Edge Detection + Watershed)
# 
# Además del filtro Moving Window, desarrollamos una técnica más sofisticada llamada **CEWS** (Canny Edge Detection + Watershed) para tratar específicamente con el efecto "salt and pepper" en las predicciones.
# 
# #### Objetivo de CEWS
# 
# - **Eliminar efecto salt and pepper**: Reducir píxeles aislados de diferentes clases que generan granularidad
# - **Segmentación inteligente**: Identificar regiones homogéneas mediante detección de bordes y segmentación
# - **Preservar estructura espacial**: Mantener los límites reales entre diferentes tipos de cultivos
# 
# #### Cómo Funciona CEWS
# 
# El algoritmo CEWS sigue estos pasos:
# 
# 1. **Normalización**: Normaliza el mapa de predicción para procesamiento
# 2. **Detección de bordes (Canny)**: Identifica los bordes entre diferentes clases usando el algoritmo de detección de bordes Canny
# 3. **Operaciones morfológicas**: Aplica cierre y apertura morfológica para limpiar y conectar regiones
# 4. **Segmentación (Watershed)**: Utiliza el algoritmo Watershed para segmentar la imagen en regiones homogéneas
# 5. **Remapeo de clases**: Para cada segmento identificado, asigna la clase mayoritaria dentro de ese segmento
# 
# Este enfoque es más sofisticado que el Moving Window porque no solo considera el vecindario inmediato, sino que identifica regiones completas y las homogeniza según la clase predominante.
# 
# #### Scripts que Desarrollamos
# 
# - `Joan/02_PruebaCEWS_3Arroyos_ModeloRF_Pixel.ipynb`: Notebook que implementa CEWS para Tres Arroyos
# - `Joan/03_PruebaCEWS_CorDiaz_ModeloRF_Pixel.ipynb`: Notebook que implementa CEWS para Coronel Suárez
# 
# ### Visualización: Predicción Original vs Suavizada (Moving Window)
# 
# A continuación comparamos las predicciones antes y después de aplicar el filtro Moving Window para ver el efecto:

# %%
# Visualizar prediccion original vs suavizada
prediccion_original = os.path.join(DATA_PROC_DIR, "11_prediccion_rf_verano.tif")
prediccion_suavizada = os.path.join(DATA_PROC_DIR, "11_prediccion_rf_verano_MW_3x3.tif")

if os.path.exists(prediccion_original) and os.path.exists(prediccion_suavizada):
    # Ajuste de memoria: reducir tamaño para visualizacion
    W_preview, H_preview = 1000, 1000
    
    with rasterio.open(prediccion_original) as src_orig, \
         rasterio.open(prediccion_suavizada) as src_suav:
        
        extent = [src_orig.bounds.left, src_orig.bounds.right, 
                 src_orig.bounds.bottom, src_orig.bounds.top]
        
        # Leer rasters con resampling para evitar problemas de memoria
        pred_orig = src_orig.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        pred_suav = src_suav.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        )
        
        # Colores para las clases
        colores_clases = {
            0: '#339820',  # Verde para CULTIVO AGRICOLA
            1: '#646b63',  # Gris para BARBECHO
            2: '#e6f0c2'   # Beige para NO AGRICOLA
        }
        cmap_custom = ListedColormap([colores_clases[0], colores_clases[1], colores_clases[2]])
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Prediccion original
        ax = axes[0]
        # Convertir a int32 para poder asignar valores negativos
        pred_orig_display = pred_orig.copy().astype(np.int32)
        pred_orig_display[pred_orig < 0] = -1
        im1 = ax.imshow(pred_orig_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Prediccion Original (Random Forest)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Prediccion suavizada
        ax = axes[1]
        # Convertir a int32 para poder asignar valores negativos
        pred_suav_display = pred_suav.copy().astype(np.int32)
        pred_suav_display[pred_suav < 0] = -1
        im2 = ax.imshow(pred_suav_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Prediccion Suavizada (Moving Window 3x3)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Leyenda comun
        nombres_clases = ['CULTIVO AGRICOLA', 'BARBECHO', 'NO AGRICOLA']
        handles = [Patch(facecolor=colores_clases[i], label=nombres_clases[i]) for i in range(3)]
        fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.02))
        
        plt.suptitle('Efecto del Filtro Moving Window 3x3', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.show()
        
        # Calcular cambios
        mask_valido = (pred_orig >= 0) & (pred_suav >= 0)
        cambios = (pred_orig[mask_valido] != pred_suav[mask_valido]).sum()
        total = mask_valido.sum()
        porcentaje_cambios = 100 * cambios / total if total > 0 else 0
        
        print(f"Pixeles modificados por el filtro: {cambios:,} de {total:,} ({porcentaje_cambios:.2f}%)")
else:
    print("Archivos de prediccion no encontrados")

# %% [markdown]
# ## Paso 7: Validación en Nuevas Zonas (Generalización)
# 
# Para evaluar si nuestro modelo realmente generaliza bien y no está sobreajustado a Tres Arroyos, decidimos aplicarlo a una nueva región: **Coronel Suárez**. Esto nos permite verificar si el modelo puede funcionar en otras áreas con características similares.
# 
# ### Proceso de Validación que Seguimos
# 
# 1. **Descarga de datos**: Descargamos imágenes Sentinel-2 para Coronel Suárez (mismo período temporal)
# 2. **Procesamiento**: Generamos características multitemporales con el mismo formato que usamos para Tres Arroyos
# 3. **Aplicación del modelo**: Usamos el modelo entrenado en Tres Arroyos (sin reentrenarlo) para generar predicciones
# 4. **Evaluación**: Comparamos las predicciones con los datos de referencia del INTA para Coronel Suárez
# 
# ### Scripts que Desarrollamos
# 
# - `scripts/Coronel Suarez/0_get_rasters_coronel_suarez.py`: Descarga de imágenes para la nueva región
# - `scripts/Coronel Suarez/11_NDVI_coronel_suarez_verano.py`: Generación de características en el mismo formato
# - `scripts/Coronel Suarez/20_test_coronel_suarez.ipynb`: Test y evaluación del modelo en la nueva zona
# 
# ### Visualización de Predicciones en Coronel Suárez
# 
# A continuación mostramos las predicciones que obtuvimos al aplicar nuestro modelo en la nueva región:

# %%
# Visualizar predicciones en Coronel Suarez: Comparacion completa con post-procesamiento
# En el raster 20_predicciones_rf_coronel_suarez.tif:
#   - Banda 1: realidad (INTA) ya agrupada en clases 0,1,2
#   - Banda 2: prediccion del Random Forest (clases 0,1,2)
prediccion_cs = os.path.join(DATA_PROC_DIR, "20_predicciones_rf_coronel_suarez.tif")
prediccion_cews_cs = os.path.join("..", "Joan", "proc_joan", "20_predicciones_rf_coronel_suarez_cews.tif")
prediccion_mwm_cs = os.path.join(DATA_PROC_DIR, "20_predicciones_rf_coronel_suarez_MW_3x3.tif")

if os.path.exists(prediccion_cs):
    # Ajuste de memoria: reducir tamaño para visualizacion
    W_preview, H_preview = 1000, 1000
    
    with rasterio.open(prediccion_cs) as src_pred:
        
        extent = [src_pred.bounds.left, src_pred.bounds.right, 
                 src_pred.bounds.bottom, src_pred.bounds.top]
        
        # Leer rasters con resampling para evitar problemas de memoria
        # Banda 1: realidad (INTA) ya en clases 0,1,2; Banda 2: prediccion RF (0,1,2)
        realidad = src_pred.read(
            1,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        ).astype(np.int32)
        prediccion_orig = src_pred.read(
            2,
            out_shape=(H_preview, W_preview),
            resampling=rasterio.enums.Resampling.nearest
        ).astype(np.int32)
        
        # Normalizamos nodata a -1 en la realidad agrupada
        realidad_agrupada = realidad.copy()
        realidad_agrupada[realidad_agrupada < 0] = -1
        
        # Aplicar MWM filter sobre el raster completo SOLO la primera vez
        if not os.path.exists(prediccion_mwm_cs):
            print("Aplicando filtro Moving Window 3x3 sobre raster completo...")

            def moda(vecindario):
                vec = vecindario[vecindario >= 0]
                if len(vec) == 0:
                    return -1
                valores, conteos = np.unique(vec, return_counts=True)
                return valores[np.argmax(conteos)]

            # Leer banda 2 (prediccion original) a resolucion completa
            prediccion_full = src_pred.read(2).astype(np.int32)

            prediccion_mwm_full = generic_filter(
                prediccion_full,
                function=moda,
                size=3,
                mode='nearest'
            ).astype(np.int32)

            # Guardar resultado en disco para reutilizar en corridas futuras
            meta_mwm = src_pred.meta.copy()
            meta_mwm.update(count=1, dtype="int32", nodata=-1)
            with rasterio.open(prediccion_mwm_cs, "w", **meta_mwm) as dst:
                dst.write(prediccion_mwm_full, 1)

            print(f"Raster Moving Window guardado en: {prediccion_mwm_cs}")

        # Para la visualizacion, leer siempre desde el raster guardado (con resampling)
        with rasterio.open(prediccion_mwm_cs) as src_mwm:
            prediccion_mwm = src_mwm.read(
                1,
                out_shape=(H_preview, W_preview),
                resampling=rasterio.enums.Resampling.nearest
            )
        
        # Cargar CEWS si existe
        if os.path.exists(prediccion_cews_cs):
            with rasterio.open(prediccion_cews_cs) as src_cews:
                prediccion_cews = src_cews.read(
                    1,
                    out_shape=(H_preview, W_preview),
                    resampling=rasterio.enums.Resampling.nearest
                )
            tiene_cews = True
        else:
            prediccion_cews = None
            tiene_cews = False
        
        # Colores para las clases
        colores_clases = {
            0: '#339820',  # Verde para CULTIVO AGRICOLA
            1: '#646b63',  # Gris para BARBECHO
            2: '#e6f0c2'   # Beige para NO AGRICOLA
        }
        cmap_custom = ListedColormap([colores_clases[0], colores_clases[1], colores_clases[2]])
        
        # Crear figura 2x2
        if tiene_cews:
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        nombres_clases = ['CULTIVO AGRICOLA', 'BARBECHO', 'NO AGRICOLA']
        
        # Preparar datos para visualizacion
        realidad_display = realidad_agrupada.copy().astype(np.int32)
        realidad_display[realidad_agrupada < 0] = -1
        
        prediccion_orig_display = prediccion_orig.copy().astype(np.int32)
        prediccion_orig_display[prediccion_orig < 0] = -1
        
        prediccion_mwm_display = prediccion_mwm.copy().astype(np.int32)
        prediccion_mwm_display[prediccion_mwm < 0] = -1
        
        if tiene_cews:
            prediccion_cews_display = prediccion_cews.copy().astype(np.int32)
            prediccion_cews_display[prediccion_cews < 0] = -1
        
        # Plot 1: Realidad INTA
        ax = axes[0, 0]
        im1 = ax.imshow(realidad_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Realidad (INTA Verano - Coronel Suarez)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Plot 2: Prediccion Original
        ax = axes[0, 1]
        im2 = ax.imshow(prediccion_orig_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Prediccion Original (Random Forest)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Plot 3: Prediccion MWM
        ax = axes[1, 0]
        im3 = ax.imshow(prediccion_mwm_display, cmap=cmap_custom, vmin=0, vmax=2, 
                       interpolation='nearest', extent=extent, origin='upper')
        ax.set_title('Prediccion Post-procesada (Moving Window 3x3)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Plot 4: Prediccion CEWS o placeholder
        ax = axes[1, 1]
        if tiene_cews:
            im4 = ax.imshow(prediccion_cews_display, cmap=cmap_custom, vmin=0, vmax=2, 
                           interpolation='nearest', extent=extent, origin='upper')
            ax.set_title('Prediccion Post-procesada (CEWS)', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'CEWS no disponible', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes)
            ax.set_title('Prediccion Post-procesada (CEWS)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Leyenda comun
        handles = [Patch(facecolor=colores_clases[i], label=nombres_clases[i]) for i in range(3)]
        fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.02))
        
        # Titulo general con mayor margen respecto a los titulos de cada subplot
        plt.suptitle('Comparacion Completa: INTA vs Predicciones Originales y Post-procesadas - Coronel Suarez', 
                     fontsize=16, fontweight='bold', y=0.98)
        # Dejamos espacio en la parte superior (top=0.9) para separar el suptitle de los ejes
        plt.tight_layout(rect=[0, 0.08, 1, 0.9])
        plt.subplots_adjust(bottom=0.08)
        plt.show()
else:
    print("Archivos de Coronel Suarez no encontrados")

# %% [markdown]
# ## Análisis Comparativo de Post-procesamiento en Coronel Suárez
# 
# En la sección anterior mostramos una comparación completa entre las predicciones originales y las versiones post-procesadas (Moving Window 3x3 y CEWS) para la región de Coronel Suárez. Este análisis nos permite evaluar el impacto de las técnicas de post-procesamiento en la calidad de las predicciones.
# 
# ### Resultados del Post-procesamiento
# 
# Los resultados obtenidos muestran cómo las técnicas de post-procesamiento afectan la accuracy del modelo:
# 
# - **Predicción Original**: Accuracy base del modelo Random Forest sin post-procesamiento
# - **Moving Window 3x3**: Aplica un filtro de mayoría en una ventana 3x3, suavizando píxeles aislados
# - **CEWS (Canny Edge Detection + Watershed)**: Segmentación avanzada que identifica regiones homogéneas y las homogeniza
# 
# ### Interpretación de los Resultados
# 
# El análisis comparativo permite evaluar:
# 
# 1. **Mejora en coherencia espacial**: Ambas técnicas de post-procesamiento mejoran la coherencia visual de las predicciones, eliminando el efecto "salt and pepper"
# 2. **Impacto en accuracy**: El post-procesamiento puede mejorar o mantener la accuracy, dependiendo de si los píxeles corregidos estaban originalmente mal clasificados
# 3. **Trade-off entre suavizado y detalle**: El post-procesamiento puede suavizar demasiado y perder detalles finos en los bordes entre clases
# 
# ### Conclusiones del Análisis
# 
# Basándonos en los resultados obtenidos:
# 
# - **Moving Window 3x3** es una técnica simple y eficiente que mejora la coherencia espacial con un impacto mínimo en el rendimiento computacional
# - **CEWS** es una técnica más sofisticada que puede preservar mejor los bordes reales entre diferentes tipos de cultivos, aunque requiere más procesamiento
# - Ambas técnicas pueden aplicarse según las necesidades específicas del proyecto
# 
# La elección entre una u otra técnica depende del balance entre:
# - **Precisión espacial**: CEWS puede preservar mejor los bordes reales
# - **Eficiencia computacional**: Moving Window es más rápido
# - **Contexto de aplicación**: Para mapas de cultivos a gran escala, Moving Window puede ser suficiente, mientras que para análisis detallados, CEWS puede ser preferible
# 
# ## Resumen del Pipeline Completo
# 
# ### Flujo de Trabajo que Desarrollamos
# 
# 1. Descarga de imágenes Sentinel-2 desde Google Earth Engine.  
# 2. Integración con datos INTA (MNC invierno y verano como verdad terreno).  
# 3. Procesamiento de características multitemporales (estadísticas NDVI y series temporales).  
# 4. Entrenamiento del modelo Random Forest con validación espacial por bloques y 3 clases.  
# 5. Predicciones del modelo y generación de mapas de clasificación.  
# 6. Post-procesamiento (Moving Window 3x3 y CEWS) para mejorar coherencia espacial y reducir ruido.  
# 7. Validación en nuevas zonas (Coronel Suárez) para evaluar la generalización del modelo.  
# 
# %%
# (Se removió la visualización con Graphviz para evitar dependencias del sistema)
# %% [markdown]
# ### Archivos Principales que Generamos
# 
# - `5_NDVI_combinado.tif`: Raster combinado de todas las imágenes NDVI mensuales
# - `9_NDVI_con_recortes.tif`: NDVI combinado con recortes MNC alineados
# - `11_NDVI_inta_verano.tif`: Raster final con características para entrenamiento (12 bandas)
# - `11_rf_model_50_estimators.pkl`: Modelo Random Forest entrenado que guardamos
# - `11_prediccion_rf_verano.tif`: Predicciones del modelo para Tres Arroyos
# - `11_prediccion_rf_verano_MW_3x3.tif`: Predicciones suavizadas con Moving Window
# - `11_prediccion_rf_verano_cews.tif`: Predicciones suavizadas con CEWS para Tres Arroyos (en `Joan/proc_joan/`)
# - `20_predicciones_rf_coronel_suarez.tif`: Predicciones en nueva zona (Coronel Suárez)
# - `20_predicciones_rf_coronel_suarez_cews.tif`: Predicciones suavizadas con CEWS para Coronel Suárez (en `Joan/proc_joan/`)
# 
# ### Resultados Principales que Obtuvimos
# 
# - **Accuracy en Tres Arroyos**: 86.27% (buen rendimiento general)
# - **Clases mejor clasificadas**: CULTIVO AGRÍCOLA (90.2% precision) - la más importante para nuestro objetivo
# - **Modelo generalizable**: Validado en nueva región (Coronel Suárez) con resultados consistentes
# - **Post-procesamiento efectivo**: 
#   - **Moving Window 3x3**: Mejora la coherencia espacial eliminando píxeles aislados (efecto salt and pepper)
#   - **CEWS**: Segmentación avanzada que preserva mejor los bordes reales entre clases
#   - Análisis comparativo en Coronel Suárez muestra el impacto de ambas técnicas en accuracy y coherencia visual
# 
# ### Consideraciones Técnicas Importantes que Tuvimos en Cuenta
# 
# 1. **Autocorrelación espacial**: Implementamos validación por bloques espaciales para evitar sobreestimación del rendimiento
# 2. **Procesamiento optimizado**: Usamos procesamiento por ventanas para manejar rasters grandes sin saturar la memoria
# 3. **Proyección consistente**: Mantuvimos UTM Zone 21S en todas las imágenes para asegurar alineación espacial correcta
# 4. **Datos multitemporales**: Las series temporales NDVI capturan patrones fenológicos que son clave para distinguir tipos de cultivos
#
# ### Principales dificultades y cómo las resolvimos
#
# - **Autocorrelación espacial en el entrenamiento y validación**: 
#   - Problema: las métricas iniciales del modelo estaban sobreestimadas porque los píxeles de entrenamiento y test estaban demasiado cerca entre sí.
#   - Solución: implementamos una validación espacial por bloques 3x3 tipo tablero de ajedrez, separando explícitamente las zonas de entrenamiento y test.
# - **Manejo de grandes volúmenes de datos raster**: 
#   - Problema: los rasters multibanda y multitemporales no eran manejables en memoria con lectura directa completa.
#   - Solución: adoptamos lectura con `rasterio` usando resampling para visualización y procesamiento por ventanas para los pasos más pesados, reduciendo el uso de memoria.
# - **Interpretación de la clase “Barbecho” en serie temporal**:
#   - Problema: los píxeles etiquetados como barbecho cambiaban de uso entre estaciones, generando firmas NDVI difíciles de interpretar si se usaba todo el año completo.
#   - Solución: restringimos el análisis y entrenamiento a la ventana de verano, donde las etiquetas INTA son más coherentes con el ciclo del cultivo que queremos modelar.
# - **Ruido tipo “salt and pepper” en las predicciones por píxel**:
#   - Problema: el mapa de predicciones presentaba píxeles aislados sin coherencia espacial, poco interpretables agronómicamente.
#   - Solución: desarrollamos un filtro Moving Window 3x3 y un esquema CEWS (Canny + Watershed) que suavizan las predicciones preservando los bordes relevantes.
#
# ### Líneas de trabajo futuro y modelo de mezcla
#
# Además del esquema supervisado con Random Forest, exploramos un enfoque no supervisado basado en **modelos de mezcla Gaussiana (GMM)**, documentado en `scripts/modelo_mezcla/modelo_mezclak3.py` y en el notebook `scripts/modelo_mezcla/mezcla_vs_inta.ipynb`.
#
# - **Modelo de mezcla GMM k=3 sobre 11 bandas NDVI**:
#   - Aplicamos un GMM con `k=3` componentes sobre las once bandas continuas (estadísticas + serie temporal NDVI) del raster `11_NDVI_inta_verano.tif`.
#   - El modelo genera un raster de clusters (`11_NDVI_inta_verano_gmm_k3(n_ver).tif`) que segmenta el paisaje en tres tipos principales según su firma fenológica.
# - **Análisis comparativo GMM vs INTA**:
#   - En el notebook `mezcla_vs_inta.ipynb` comparamos los clusters GMM con las clases INTA mediante histogramas, PCA y tablas de contingencia.
#   - Los resultados sugieren que los tres clusters se alinean de forma aproximada con las categorías “barbecho”, “agrícola” y “no agrícola”, aportando una segmentación no supervisada coherente con la clasificación oficial.
# - **Pasos futuros basados en este modelo de mezcla**:
#   - Integrar el GMM como módulo de pre-clasificación o segmentación para reducir ruido y definir unidades relativamente homogéneas antes del modelo supervisado.
#   - Ajustar el número de clusters (`k`) y evaluar si una segmentación más fina ayuda a distinguir subtipos de cultivos dentro de la clase agrícola.
#   - Extender el análisis a Coronel Suárez y a otras regiones para evaluar si la estructura de clusters se mantiene estable espacialmente.
#   - Explorar esquemas híbridos donde el GMM aporte información adicional (cluster ID, probabilidades de pertenencia) como features para modelos supervisados más complejos.
#

# %%
