import numpy as np
import rasterio
from scipy.ndimage import generic_filter

# Ruta del raster de predicción RF
raster_path = r"C:\Users\eugea\Downloads\11_prediccion_rf_verano.tif"

# ---------------------------
# Función de moda para ventana
# ---------------------------
def moda(vecindario):
    vec = vecindario[vecindario >= 0]  # ignorar nodata
    if len(vec) == 0:
        return -1
    valores, conteos = np.unique(vec, return_counts=True)
    return valores[np.argmax(conteos)]

# ---------------------------
# Leer raster
# ---------------------------
with rasterio.open(raster_path) as src:
    pred = src.read(1)
    meta = src.meta.copy()

# ---------------------------
# Aplicar Moving Window (3x3)
# ---------------------------
print("Aplicando filtro de mayoría (3x3)...")

pred_filtrado = generic_filter(
    pred,
    function=moda,
    size=3,           # ventana 3×3 (cambiar si queiro de 5 o 7)
    mode='nearest'    # bordes
)

print("Filtro aplicado correctamente.")

# ---------------------------
# Guardar resultado
# ---------------------------
output_path = r"C:\Users\eugea\Downloads\11_prediccion_rf_verano_moving_window.tif"

meta.update({
    'dtype': 'int32',
    'count': 1,
    'nodata': -1
})

with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(pred_filtrado.astype(np.int32), 1)

print(f"Raster suavizado guardado en:\n  {output_path}")

# import os
# import numpy as np
# import rasterio
# from scipy.ndimage import generic_filter

# ---------------------------
# Paths del proyecto
# ---------------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# PROC_DIR = os.path.join(DATA_DIR, "proc")

# # Raster de entrada: predicción RF verano
# raster_path = os.path.join(PROC_DIR, "11_prediccion_rf_verano.tif")

# # Raster de salida (moving window)
# output_path = os.path.join(PROC_DIR, "11_prediccion_rf_verano_MW_3x3.tif")

# ---------------------------
# Función de moda (mayoría)
# ---------------------------
# def moda(vecindario):
#     vec = vecindario[vecindario >= 0]  # ignorar nodata
#     if len(vec) == 0:
#         return -1
#     valores, conteos = np.unique(vec, return_counts=True)
#     return valores[np.argmax(conteos)]


# # ---------------------------
# # Leer raster
# # ---------------------------
# print("Leyendo raster de predicción RF...")
# with rasterio.open(RASTER_PATH) as src:
#     pred = src.read(1)
#     meta = src.meta.copy()


# # ---------------------------
# # Aplicar Moving Window 3×3
# # ---------------------------
# print("Aplicando filtro de mayoría (3x3)...")

# pred_filtrado = generic_filter(
#     pred,
#     function=moda,
#     size=3,
#     mode='nearest'
# )

# print("Filtro aplicado correctamente.")


# # ---------------------------
# # Guardar raster suavizado
# # ---------------------------
# meta.update({
#     'dtype': 'int32',
#     'count': 1,
#     'nodata': -1
# })

# with rasterio.open(output_path, 'w', **meta) as dst:
#     dst.write(pred_filtrado.astype(np.int32), 1)

# print(f"Raster suavizado guardado en:\n  {output_path}")
