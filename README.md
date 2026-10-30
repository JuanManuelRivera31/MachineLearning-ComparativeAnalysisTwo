# Análisis comparativo de modelos de Machine Learning

Segunda entrega de **Inteligencia Computacional II** (Universidad de la Amazonia, 2026-1).

El proyecto amplía los modelos de la primera entrega (Árbol de Decisión, Random Forest y KNN) con modelos bayesianos, máquinas de soporte vectorial y redes neuronales. Todos se comparan con los mismos datos, la misma partición y el mismo conjunto TEST, en dos problemas:

| Problema | Datos | Registros | Variable objetivo | Modelos nuevos |
|---|---|---|---|---|
| **Regresión** | Biometría de jugadores de e-sports | 250 000 | `Reaction_Time_ms` (tiempo de reacción) | Regresión lineal bayesiana, SVR y 3 redes neuronales |
| **Clasificación** | Reservas de hotel | 119 390 | `is_canceled` (cancelación de la reserva) | Naive Bayes, Red Bayesiana, SVM y 3 redes neuronales |

El detalle de la metodología y el análisis está en el [informe en PDF](Informe_Entrega2.pdf).

## Resultados

Las tablas muestran el desempeño sobre TEST. Las cifras provienen de [`e2_tabla_04`](base1/resultados_entrega2/e2_tabla_04_comparacion_final_test.csv) y [`e2_tabla_07`](base2/resultados_entrega2/e2_tabla_07_comparacion_final_test.csv).

### Regresión

| Modelo | Entrega | RMSE (ms) | R² | Entrenamiento (s) |
|---|---|---:|---:|---:|
| Red neuronal grande (128, 128, 128) | 2 | 14.9956 | 0.3784 | 43.06 |
| Red neuronal suave (16) | 2 | 14.9960 | 0.3784 | 3.87 |
| Red neuronal intermedia (64, 64) | 2 | 14.9983 | 0.3782 | 10.26 |
| Regresión bayesiana | 2 | 15.0609 | 0.3730 | 0.65 |
| SVR lineal (C=1, ε=5) | 2 | 15.0617 | 0.3729 | 0.83 |
| Random Forest podado | 1 | 15.1495 | 0.3656 | 2.62 |
| KNN (K=50) | 1 | 15.2923 | 0.3536 | 0.62 |
| Árbol podado | 1 | 15.3323 | 0.3502 | 0.61 |
| Baseline (media) | Referencia | 19.0207 | 0.0000 | 0.10 |

Los cinco modelos nuevos superan a los de la primera entrega. Las diferencias son pequeñas y el tamaño de la red casi no cambia el error, por lo que la red suave y la regresión bayesiana son las opciones más razonables por su bajo costo.

### Clasificación

| Modelo | Entrega | Umbral | F1 | ROC-AUC | Entrenamiento (s) |
|---|---|---:|---:|---:|---:|
| Random Forest ajustado | 1 | 0.49 | 0.7951 | 0.9285 | 29.53 |
| Red neuronal intermedia (64, 64) | 2 | 0.38 | 0.7870 | 0.9212 | 5.36 |
| Red neuronal grande (128, 128, 128) | 2 | 0.34 | 0.7815 | 0.9192 | 12.98 |
| Red neuronal suave (16) | 2 | 0.38 | 0.7779 | 0.9173 | 2.63 |
| SVM RBF (C=1) | 2 | 0.04 | 0.7741 | 0.9056 | 49.33 |
| Árbol podado | 1 | 0.58 | 0.7708 | 0.9128 | 2.72 |
| KNN ajustado | 1 | 0.43 | 0.7592 | 0.9014 | 0.53 |
| Red Bayesiana (11 atributos) | 2 | 0.41 | 0.7531 | 0.8993 | 2.09 |
| Naive Bayes categórico | 2 | 0.28 | 0.6884 | 0.8308 | 0.55 |
| Baseline (clase mayoritaria) | Referencia | 0.50 | 0.0000 | 0.5000 | 0.00 |

Random Forest sigue en primer lugar, seguido de las redes neuronales, que entrenan en bastante menos tiempo. La Red Bayesiana supera a Naive Bayes porque aprende las dependencias entre atributos en lugar de suponer que son independientes.

## Estructura

```text
.
├── Informe_Entrega2.pdf        Informe final (también en .docx)
├── requirements.txt            Dependencias del proyecto
├── base1/                      Regresión
│   ├── Entrega2_Base1_Regresion_Bayes_SVM_RedNeuronal.ipynb
│   ├── data/                   Datos de entrada
│   ├── entrega1_resultados/    Resultados de la Entrega 1, usados para comparar
│   └── resultados_entrega2/    Tablas y figuras generadas
└── base2/                      Clasificación
    ├── Entrega2_Base2_Clasificacion_Bayes_SVM_RedNeuronal.ipynb
    ├── helpers.py              Métricas, preprocesadores y transformadores
    ├── data/
    ├── entrega1_resultados/
    └── resultados_entrega2/
```

## Cómo ejecutarlo

El proyecto se desarrolló con **Python 3.14**. Desde la raíz del repositorio, en Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Después abre el notebook de `base1/` o `base2/`, selecciona el kernel de `.venv` y ejecuta todas las celdas. Cada notebook debe ejecutarse desde su propia carpeta, porque lee los datos con rutas relativas.

Los parámetros del experimento están en la celda de configuración de cada notebook: semilla `RANDOM_STATE = 42`, partición Hold-Out 70/30 y validación cruzada de 5 folds. Al ejecutarlos, las tablas y figuras se guardan en `resultados_entrega2/`.

## Autores

- Daniel Felipe Sánchez Perafan
- Juan Manuel Rivera
- Nestor David Torres Rojas

Docente: Jesús Emilio Pinto Lopera. Programa de Ingeniería de Sistemas, Universidad de la Amazonia.
