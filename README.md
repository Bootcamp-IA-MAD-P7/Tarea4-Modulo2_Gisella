# Tarea4-Modulo2_Gisella

Tarea : Investigación y Desarrollo sobre Modelos Ensemble

# Tarea: Investigación y Desarrollo sobre Modelos Ensemble

Deben investigar cómo funcionan los modelos ensemble, sus principales tipos, ventajas,
desventajas y aplicaciones en el mundo real.

Parte 1: Modelos Ensemble en Machine Learning Supervisado
Cada estudiante debe investigar y responder las siguientes preguntas de forma detallada:

1. ¿Qué es un modelo ensemble en Machine Learning y cuál es su propósito? ¿En qué se basan para mejorar el rendimiento respecto a un modelo individual?
   Definición y Propósito
   Un modelo ensemble (o aprendizaje por conjuntos) es una técnica de Machine Learning en la que se combinan las predicciones de múltiples modelos individuales (llamados modelos base o learners débiles) para producir una predicción final unificada que resulta ser más precisa y robusta que la de cualquiera de los modelos por separado.
   El propósito principal de un modelo ensemble es maximizar el rendimiento predictivo y reducir el error general del sistema de aprendizaje automático. En lugar de confiar en la decisión de un solo algoritmo (como un único árbol de decisión), un ensemble consulta a un "comité" de modelos y toma una decisión basada en el consenso (mediante votación, promedios o ponderaciones).
   ¿En qué se basan para mejorar el rendimiento respecto a un modelo individual?
   Los modelos ensemble basan su éxito principalmente en tres pilares y principios teóricos:
   La "Sabiduría de la Multitud" (Wisdom of the Crowd):
   Se fundamenta en el principio matemático de que la combinación de múltiples estimadores con errores no correlacionados reduce el error global. Si los modelos individuales cometen errores en diferentes casos o regiones del espacio de datos, al promediar o votar, los errores individuales se cancelan entre sí.

Descomposición del Error: Sesgo, Varianza y Ruido (Bias-Variance Tradeoff): El error total de un modelo se divide en tres componentes: Sesgo (Bias), Varianza (Variance) y Ruido irreducible. Los modelos ensemble mejoran el rendimiento atacando directamente estos componentes:
Reducción de la Varianza (evitar Overfitting): Métodos como Bagging (ej. Random Forest) entrenan múltiples modelos complejos e independientes. Al promediar sus predicciones, se reduce drásticamente la variabilidad del modelo frente a nuevos datos sin aumentar el sesgo.
Reducción del Sesgo (evitar Underfitting): Métodos como Boosting (ej. XGBoost, AdaBoost) combinan modelos muy simples que individualmente tienen un alto sesgo. Al entrenarlos de manera secuencial, donde cada nuevo modelo corrige los errores del anterior, se logra reducir progresivamente el sesgo del conjunto.

Diversidad de los Modelos Base:
Para que un ensemble funcione eficazmente, los modelos que lo componen deben ser diversos (deben cometer errores en diferentes tipos de datos). La diversidad se logra entrenando modelos con diferentes subconjuntos de datos, diferentes conjuntos de características (features) o utilizando distintos algoritmos de aprendizaje.

## 2. Explique la diferencia conceptual entre Bagging y Boosting, mencionando un algoritmo representativo de cada técnica.

Tanto Bagging (Bootstrap Aggregating) como Boosting son metodologías de ensamblado que combinan múltiples modelos base para mejorar la precisión, pero difieren fundamentalmente en cómo entrenan los modelos y cuál es su objetivo principal de optimización.

Diferencias Conceptuales Principales

Característica
Bagging (Bootstrap Aggregating)
Boosting
Construcción de Modelos
Paralela e Independiente: Todos los modelos base se entrenan al mismo tiempo y de forma independiente con diferentes submuestras de datos.
Secuencial y Dependiente: Los modelos se entrenan uno tras otro. Cada nuevo modelo se enfoca en corregir los errores cometidos por los modelos anteriores.
Muestreo de Datos
Utiliza Bootstrapping (muestreo aleatorio con reemplazo) asignando el mismo peso a cada observación de entrenamiento.
Ajusta los pesos de los datos: las observaciones clasificadas erróneamente en una etapa reciben un mayor peso para la siguiente etapa.

Votación simple/Promedio: Todas las predicciones de los modelos base tienen el mismo peso en la decisión final.
Votación Ponderada: Las predicciones de los modelos con mayor precisión/rendimiento tienen mayor peso en la decisión final.
Objetivo Principal
Reducir la Varianza (evitar Overfitting): Ideal para modelos complejos e inestables que tienden a memorizar el ruido de los datos.
Reducir el Sesgo (evitar Underfitting): Ideal para modelos simples (learners débiles) que necesitan mejorar su capacidad predictiva.

Algoritmos Representativos
Algoritmo representativo de Bagging: Random Forest (Bosques Aleatorios)
Funcionamiento: Entrena múltiples árboles de decisión complejos en paralelo utilizando submuestras aleatorias de los datos y de las características (features). Luego, promedia sus predicciones (en regresión) o realiza una votación por mayoría (en clasificación).
Algoritmo representativo de Boosting: XGBoost (eXtreme Gradient Boosting) / AdaBoost (Adaptive Boosting)
Funcionamiento: Entrena una serie de árboles de decisión muy simples (árboles poco profundos) de manera secuencial. Cada nuevo árbol se ajusta a los residuos/errores calculados por el árbol anterior utilizando descenso de gradiente.

## 3. ¿En qué consiste el Stacking como técnica de ensemble y cómo difiere su enfoque de Bagging y Boosting?

¿En qué consiste el Stacking (Stacked Generalization)?
El Stacking es una técnica de ensamblado en la que se entrenan varios modelos diferentes de Machine Learning (llamados Modelos Base o Nivel 0) y, en lugar de combinar sus predicciones mediante una regla fija (como el promedio o la votación por mayoría), se entrena un nuevo modelo de Machine Learning (llamado Meta-Modelo o Nivel 1) para que aprenda cuál es la mejor forma de combinar las predicciones de los modelos base.
El proceso funciona en dos niveles:
Nivel 0 (Modelos Base): Se entrenan múltiples modelos con los datos originales. Estos modelos suelen ser heterogéneos (por ejemplo, una Regresión Logística, un Random Forest y una Red Neuronal).

Nivel 1 (Meta-Modelo): Se utiliza la salida o predicciones generadas por los modelos del Nivel 0 como características de entrada (features) para entrenar un modelo final (por ejemplo, una Regresión Lineal o Lasso), el cual aprende qué modelo del Nivel 0 es más confiable para cada tipo de dato y realiza la predicción definitiva.
¿Cómo difiere el enfoque de Stacking de Bagging y Boosting?
Las principales diferencias entre Stacking y las otras dos técnicas se resumen en tres aspectos fundamentales:

Aspecto  
Bagging
Boosting
Stacking
Diversidad de Modelos
Homogéneos: Utiliza múltiples instancias del mismo tipo de algoritmo (habitualmente Árboles de Decisión).
Homogéneos: Utiliza múltiples instancias del mismo algoritmo base (frecuentemente árboles poco profundos).
Heterogéneos: Combina deliberadamente algoritmos de distinta naturaleza (ej. SVM, Random Forest, Gradient Boosting).
Mecanismo de Combinación
Regla Determinista/Fija: Promedio simple o votación por mayoría directa de los modelos.
Votación Ponderada: Asigna pesos a cada modelo en función de su error, pero la combinación sigue una regla predefinida.
Aprendizaje Automático (Meta-Modelo): Un algoritmo aprende mediante entrenamiento a ponderar y combinar las predicciones.
Estructura de Entrenamiento
Paralela: Todos los modelos base aprenden simultáneamente del subconjunto de datos.
Secuencial: Cada modelo aprende de los errores cometidos por el modelo anterior.
En Niveles (Hierárquica): Los modelos base entrenan con los datos y sus predicciones alimentan al meta-modelo.

## 4. ¿Por qué los modelos de ensemble suelen lograr mejor generalización que los modelos individuales, desde la perspectiva del compromiso sesgo-varianza?

La capacidad de generalización de cualquier modelo de Machine Learning (es decir, su habilidad para realizar predicciones precisas sobre datos nuevos no vistos) está determinada por el equilibrio entre el sesgo (bias) y la varianza (variance).
Un modelo individual suele sufrir de uno de dos problemas extremos:
Alto Sesgo (Underfitting): El modelo es demasiado simple y no logra capturar los patrones subyacentes de los datos.
Alta Varianza (Overfitting): El modelo es tan complejo que "memoriza" el ruido del conjunto de entrenamiento y falla al predecir datos nuevos.
Los modelos ensemble logran una mejor generalización porque permiten atacar y controlar directamente cualquiera de ambos extremos del dilema sesgo-varianza, dependiendo de la estrategia empleada:

1. Reducción de la Varianza (Estrategias tipo Bagging)
   El Problema del Modelo Individual: Algoritmos como los Árboles de Decisión profundos son modelos de alto rendimiento pero alta varianza; pequeños cambios en los datos de entrenamiento generan predicciones drásticamente diferentes.
   Solución del Ensemble: Al entrenar múltiples modelos complejos de forma independiente en diferentes subconjuntos de datos y promediar sus predicciones, la varianza del conjunto disminuye en un factor directamente proporcional al número de modelos (asumiendo cierta independencia entre ellos).
   Efecto en la Generalización: Se mantiene la capacidad del modelo individual para capturar patrones complejos (bajo sesgo), pero se elimina el ruido y la inestabilidad (baja varianza), reduciendo drásticamente el sobreajuste (overfitting).
2. Reducción del Sesgo (Estrategias tipo Boosting)
   El Problema del Modelo Individual: Los modelos extremadamente simples (learners débiles, como árboles de decisión de un solo corte) tienen una alta tasa de error por sesgo, ya que son incapaces de entender la complejidad del problema.
   Solución del Ensemble: El Boosting toma estos modelos con alto sesgo y los entrena de forma secuencial, donde cada nuevo modelo se enfoca específicamente en los residuos/errores que el modelo anterior no pudo resolver.
   Efecto en la Generalización: Al ir acumulando las correcciones de cada etapa, el conjunto reduce progresivamente el sesgo general hasta construir una frontera de decisión compleja y precisa, sin disparar descontroladamente la varianza.
   En resumen:
   Desde la perspectiva del dilema sesgo-varianza, los modelos individuales suelen verse forzados a sacrificar uno por el otro. En cambio, los modelos ensemble logran optimizar el error total ($Error = Sesgo^2 + Varianza + Ruido$) al tomar componentes con deficiencias específicas y combinarlos de forma tal que sus errores individuales se cancelen, logrando así una tasa de generalización significativamente superior.

## 5. Explicación de modelos de Ensemble

###1. AdaBoost (Adaptive Boosting)

- **Mecanismo:** Ajusta pesos a los datos en cada iteración. Entrena secuencialmente clasificadores muy simples (stumps o árboles de un solo nivel). En cada paso, aumenta el peso de las muestras clasificadas incorrectamente para que el siguiente modelo se enfoque en ellas.
- ** Ventajas:** Es fácil de implementar, computacionalmente rápido y requiere pocos hiperparámetros.
- **Casos de uso comunes:** Detección de rostros (ej. algoritmo Viola-Jones), clasificación binaria de texto y filtrado de spam.
  ###2. Gradient Boosting (Gradient Boosting Machines - GBM)
- **Mecanismo:** A diferencia de AdaBoost, no ajusta el peso de las muestras, sino que entrena cada nuevo árbol sobre los residuos (los errores directos) del conjunto de árboles anteriores utilizando el algoritmo de Descenso de Gradiente para minimizar la función de pérdida.
- **Ventajas:** Muy alta precisión y flexibilidad para trabajar con diversas funciones de pérdida (regresión, clasificación, ranking).
- **Casos de uso comunes:** Predicción de morosidad crediticia, análisis de riesgo financiero y motores de recomendación.
  ###3. XGBoost (eXtreme Gradient Boosting)
- **Mecanismo:** Es una implementación altamente optimizada y eficiente de Gradient Boosting. Utiliza regularización (L1 y L2) para evitar el sobreajuste, procesamiento en paralelo de columnas, poda de árboles previa y manejo inteligente de valores nulos.
- **Ventajas:** Excepcional velocidad de cómputo, excelente rendimiento predictivo y muy robusto frente a datos faltantes. Es el algoritmo líder en competencias de Data Science (Kaggle).
- **Casos de uso comunes:** Detección de fraudes financieros, predicción de fuga de clientes (churn), y cualquier problema de datos tabulares a gran escala.
  ###4. LightGBM (Light Gradient Boosting Machine)
- **Mecanismo:** Algoritmo desarrollado por Microsoft que utiliza una estrategia de crecimiento de árbol orientada por hojas (Leaf-wise) en lugar de nivel por nivel (Level-wise). Además, aplica las técnicas GOSS (para filtrar instancias por gradiente) y EFB (para agrupar variables dispersas).
- **Ventajas:** Extrema velocidad de entrenamiento y menor consumo de memoria manteniendo la misma o mejor precisión que XGBoost.
- **Casos de uso comunes:** Procesamiento de Big Data, análisis en tiempo real y sistemas de recomendación con millones de registros.
  ###5. CatBoost (Categorical Boosting)
- **Mecanismo:** Desarrollado por Yandex, está diseñado específicamente para tratar variables categóricas de forma nativa mediante técnicas de Target Encoding que evitan la fuga de datos (Data Leakage).
- **Ventajas:** Excelente rendimiento "de fábrica" con pocos ajustes de hiperparámetros y manejo directo de texto/categorías sin necesidad de preprocesamiento manual (como One-Hot Encoding).
- **Casos de uso comunes:** Análisis de datos bancarios, comercio electrónico, marketing predictivo y datos con alta presencia de variables cualitativas.
  ###6. Voting Classifier
- **Mecanismo:** Combina las predicciones de varios modelos de Machine Learning (que pueden ser heterogéneos) mediante una regla de votación simple: Hard Voting (elige la clase que recibe más votos) o Soft Voting (promedia las probabilidades predichas por cada modelo).
- **Ventajas:** Simple de entender e implementar; reduce sustancialmente el riesgo de que un solo modelo falle.
- **Casos de uso comunes:** Tareas de clasificación general donde se cuenta con varios modelos competitivos de diversa naturaleza (ej. SVM + Random Forest + Regresión Logística).
  ###7. Stacking Classifier
- **Mecanismo:** Utiliza un enfoque en dos niveles: entrena múltiples modelos base (Nivel 0) sobre los datos y luego utiliza sus predicciones finales como variables de entrada para un meta-modelo (Nivel 1), el cual aprende a combinar los resultados óptimamente.
- **Ventajas:** Aprovecha las fortalezas individuales de distintos algoritmos superando el rendimiento de las votaciones promedio tradicionales.
  Casos de uso comunes: Competencias de analítica avanzada donde cada décima de precisión cuenta y problemas complejos con patrones no lineales diversos.

````python# Importación de librerías
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Algoritmos de Ensemble
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Crear dataset de prueba
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entrenar e evaluar un modelo de Bagging (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Acc. Random Forest:", accuracy_score(y_test, rf_model.predict(X_test)))

# 3. Entrenar e evaluar un modelo de Boosting (AdaBoost)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
print("Acc. AdaBoost:", accuracy_score(y_test, ada_model.predict(X_test)))

# 4. Entrenar e evaluar un Voting Classifier
voting_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True))
    ],
    voting='soft'
)
voting_model.fit(X_train, y_train)
print("Acc. Voting Classifier:", accuracy_score(y_test, voting_model.predict(X_test)))```

````
