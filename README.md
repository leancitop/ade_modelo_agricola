# Modelo Agrícola ADE

Proyecto de análisis y procesamiento de imágenes satelitales para la identificación y etiquetado de segmentos agrícolas.

## Descripción

Este proyecto procesa imágenes raster (GeoTIFF) de satélites Sentinel-2 y datos del INTA para:
- Recortar y alinear imágenes raster
- Identificar campos agrícolas mediante segmentación
- Etiquetar segmentos como campo/no-campo basado en datos de referencia del INTA
- Generar visualizaciones y métricas de evaluación

## Requisitos

- Python 3.8+
- Las dependencias se encuentran en `requirements.txt`

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/ade_modelo_agricola.git
cd ade_modelo_agricola
```

2. Crea un entorno virtual:
```bash
python -m venv venv
```

3. Activa el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
ade_modelo_agricola/
├── data/                    # Datos de entrada y salida
│   ├── dataframe_region_stats.csv
│   ├── labels_ws.npy
│   └── segmentos_etiquetados.tif
├── scripts/
│   └── final.ipynb         # Notebook principal con el análisis
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## Uso

Abre el notebook `scripts/final.ipynb` en Jupyter para ejecutar el análisis:

```bash
jupyter notebook scripts/final.ipynb
```

## Dependencias Principales

- `numpy>=1.21.0` - Operaciones numéricas
- `rasterio>=1.3.0` - Procesamiento de imágenes raster
- `matplotlib>=3.5.0` - Visualización
- `scipy>=1.7.0` - Operaciones científicas
- `scikit-image>=0.18.0` - Procesamiento de imágenes

## Notas

- Asegúrate de tener los archivos de datos necesarios en la carpeta `data/`
- Los paths en el notebook pueden necesitar ajustarse según tu configuración local

## Licencia

[Especifica tu licencia aquí]

## Autores

[Agrega los nombres de los colaboradores]

