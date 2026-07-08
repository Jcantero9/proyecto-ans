# Clustering sobre el Dry Bean Dataset

Proyecto Final del módulo **Aprendizaje No Supervisado** de la Maestría en Ciencias de la Inteligencia Artificial (MSCIA), FIUNA, 2026.

Aplicación de un flujo completo de clustering sobre el Dry Bean Dataset, desde el análisis exploratorio hasta la interpretación crítica de los resultados, contrastando tres algoritmos (dos clásicos y uno avanzado) con K-Means como referencia.

> **Autor:** _(completar tu nombre)_
> **Repositorio:** _(completar la URL de GitHub)_

---

## Descripción

El objetivo es recuperar de forma **no supervisada** la estructura de 7 especies de frijol a partir de variables de forma y geometría, sin usar las etiquetas para entrenar. Como el dataset sí tiene clases reales, estas se emplean solo para la **evaluación externa** de la calidad del agrupamiento.

El proyecto se desarrolla bajo dos restricciones metodológicas deliberadas:

- **Sin PCA** en ningún paso (ni preprocesamiento ni visualización).
- **Sin DBSCAN**, para forzar la exploración de un abanico más amplio de técnicas.

Algoritmos comparados:

| Rol | Algoritmo |
|-----|-----------|
| Requerido 1 (clásico) | Gaussian Mixture Model (GMM) |
| Requerido 2 (clásico) | Spectral Clustering |
| Avanzado | Deep Clustering (autoencoder + K-Means en el espacio latente) |
| Referencia | K-Means |

---

## Dataset

**Dry Bean Dataset**, UCI Machine Learning Repository (id 602).

- 13 611 observaciones de granos de frijol.
- 16 variables numéricas de forma y geometría.
- 7 clases botánicas: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira.

**Fuente:** Koklu, M. y Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.

### Cómo obtener los datos

El dataset no se versiona en este repositorio. Se puede obtener de dos formas:

**Opción A. Descarga manual (recomendada)**

1. Descargar el dataset desde el UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
2. Descomprimir y colocar el archivo de datos dentro de `data/raw/` con el nombre que espera el loader `src/data_loading.py`.

**Opción B. Descarga programática con `ucimlrepo`**

```python
from ucimlrepo import fetch_ucirepo
dry_bean = fetch_ucirepo(id=602)
X = dry_bean.data.features
y = dry_bean.data.targets
```

> Verificá que el nombre y formato del archivo en `data/raw/` coincidan con lo que lee tu función `load_dry_bean_dataset`.

---

## Estructura del repositorio

```
.
├── data/
│   └── raw/                     # dataset (no versionado, ver instrucciones)
├── notebooks/
│   └── dry_beans_clustering.ipynb
├── src/
│   ├── data_loading.py          # carga del dataset
│   ├── eda.py                   # análisis exploratorio
│   ├── preprocessing.py         # imputación, selección por correlación, escalado
│   ├── clustering.py            # KMeans, GMM, Spectral, Deep Clustering
│   └── evaluation.py            # métricas internas/externas, matrices alineadas
├── outputs/
│   ├── figures/                 # figuras generadas (300/150 dpi)
│   └── tables/                  # tablas de resultados en CSV
├── requirements.txt
└── README.md
```

---

## Requisitos e instalación

Requiere **Python 3.10 o superior**.

```bash
# 1. Clonar el repositorio
git clone <URL-del-repositorio>
cd <nombre-del-repositorio>

# 2. Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Reproducción

1. Descargar el dataset y colocarlo en `data/raw/` (ver sección Dataset).
2. Abrir el notebook:

```bash
jupyter notebook notebooks/dry_beans_clustering.ipynb
```

3. Ejecutar las celdas en orden. El notebook corre de principio a fin sin errores y guarda automáticamente las figuras en `outputs/figures/` y las tablas en `outputs/tables/`.

**Reproducibilidad:** se fija una semilla global `RANDOM_STATE = 42` en NumPy, en el submuestreo estratificado y en t-SNE. Los componentes de aprendizaje profundo (autoencoder) pueden presentar una variación menor entre corridas según la configuración de semillas de TensorFlow.

---

## Metodología

1. **EDA.** Calidad de datos, distribuciones, correlaciones, outliers y balance de clases.
2. **Preprocesamiento.** Imputación por mediana, selección de variables por correlación de Pearson con umbral 0,95 (de 16 a 10 variables) y escalado con StandardScaler. Sin PCA.
3. **Submuestreo.** Muestra estratificada de 3 000 observaciones, reutilizada por todos los métodos para una comparación justa (Spectral Clustering escala O(n^3)).
4. **Clustering.** Búsqueda de hiperparámetros por método (GMM, Spectral, barrido de k en el espacio latente del autoencoder) y K-Means como referencia.
5. **Evaluación.** Métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz) y externas (ARI, NMI, homogeneidad, completitud, v-measure), más matrices de confusión alineadas por el algoritmo húngaro.
6. **Visualización.** Proyección t-SNE con inicialización aleatoria (sin PCA implícito).

---

## Resultados principales

| Método | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI |
|--------|:----------:|:--------------:|:-----------------:|:---:|:---:|
| K-Means (referencia) | 0.278 | 1.179 | 1267.26 | **0.665** | **0.708** |
| GMM | 0.317 | 1.436 | 1201.91 | 0.323 | 0.481 |
| Spectral Clustering | **0.472** | **0.684** | 554.48 | 0.033 | 0.163 |
| Deep Clustering | 0.199 | 1.481 | 935.56 | 0.468 | 0.552 |

**Hallazgo central:** una buena separación geométrica no equivale a capturar la taxonomía real. Spectral Clustering logra el mejor Silhouette pero el peor ARI, mientras que K-Means, el método más simple, es el más cercano a las especies reales. Deep Clustering queda en una posición intermedia, superando a GMM y Spectral en métricas externas. Bombay es la clase más fácil de recuperar por ser geométricamente muy distinta; Sira y Dermason se solapan.

---

## Limitaciones y próximos pasos

- El submuestreo de 3 000 observaciones (por el costo de Spectral Clustering) reduce el tamaño efectivo de entrenamiento.
- La dimensión del espacio latente del autoencoder se fijó de forma razonada, no por optimización exhaustiva.
- **Próximos pasos:** Deep Embedded Clustering (DEC/IDEC) con ajuste conjunto de reconstrucción y asignación de clusters, y Spectral Clustering sobre el dataset completo con aproximación espectral de Nyström.

---

## Referencias

- Koklu, M. y Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- van der Maaten, L. y Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
- von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395-416.
- Xie, J., Girshick, R. y Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis. *ICML*, 478-487.

---

## Autor y licencia

Desarrollado por José Eduardo Cantero Id para el módulo de Aprendizaje No Supervisado de la MSCIA, FIUNA, 2026.

Licencia sugerida: MIT _(agregar un archivo `LICENSE` si se desea)_.

> **GitHub Pages:** este README puede publicarse como sitio en Settings > Pages, eligiendo la rama principal y la carpeta raíz, para una versión web del proyecto.
