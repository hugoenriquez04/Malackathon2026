# Informe Tecnico — Reto 2: Feature Engineering y Preparacion de Datos para Clustering

**Proyecto:** Mercedes-Benz Hackathon — "Clustering the Star"  
**Autor:** Estudiante de Grado en Ingenieria Informatica  
**Modulo:** Reto 2 — Seleccion de variables, ingenieria de caracteristicas y pipeline de preprocesamiento  
**Fecha:** Octubre 2026  

---

## 1. Introduccion y Objetivos

El objetivo de este reto consiste en disenar, justificar e implementar el conjunto de caracteristicas (features) que serviran de entrada al modelo de aprendizaje no supervisado (clustering). 

En proyectos de ciencia de datos, alimentar un algoritmo de agrupamiento con todas las columnas disponibles de forma indiscriminada conduce a malos resultados debido a la presencia de ruido, multicolinealidad y la denominada "maldicion de la dimensionalidad". Por tanto, este trabajo aborda la preparacion de los datos desde una perspectiva de ingenieria del software y analisis de negocio:

1. Seleccionar de forma razonada que atributos aportan informacion relevante sobre el comportamiento y valor del cliente.
2. Disenar nuevas variables derivadas con significado de negocio que sinteticen patrones complejos.
3. Tratar adecuadamente las variables categoricas sin disparar artificialmente el numero de dimensiones.
4. Resolver la imputacion de valores nulos y aplicar un escalado matematicamente robusto frente a valores atipicos (outliers).
5. Encapsular todo el flujo de transformacion en un pipeline modular de Scikit-Learn que garantice la reproducibilidad.

---

## 2. Seleccion y Descarte Justificado de Variables

El dataset inicial consta de 39 columnas y mas de 302.000 registros. A continuacion se detallan las variables eliminadas y el motivo tecnico de su exclusion antes de alimentar el modelo.

### 2.1 Columnas Excluidas y Justificacion

| Columna | Tipo original | Motivo de exclusion |
| :--- | :--- | :--- |
| `Unnamed: 0` | Numerico | Indice autogenerado sin ningun valor descriptivo ni analitico. |
| `zipcode` | Categorico / Texto | Excesiva granularidad (miles de codigos postales dispersos). El entorno geografico ya queda representado de forma mas compacta por `country_code` y `population`. |
| `number_of_households` | Numerico | Presenta una correlacion lineal perfecta (r = 1.00) con `population`. En distancias euclidianas, mantener ambas variables duplicaria artificialmente el peso del tamano del municipio. |
| `purchasing_power_per_household_eur` | Numerico | Correlacion superior a 0.96 con `purchasing_power_per_inhabitant_eur`. Se descarta para evitar redundancia de escala. |
| `purchasing_power_millions_eur` | Numerico | Correlacion superior a 0.96 con el poder adquisitivo por habitante. Es una magnitud agregada municipal que no refleja directamente al cliente individual. |
| `fuel_type` | Categorico | Presenta un 59% de valores nulos y aporta informacion redundante con `engine_concept` (combustion, hibrido, electrico). |
| `speed_mode` y `trip_distance_mode` | Texto | Registros con formatos no uniformes y cadenas de texto dificiles de parsear de forma fiable. |
| `speed_average` y `trip_distance_average` | Numerico | Se descartan en favor de las medianas (`speed_median` y `trip_distance_median`), ya que los promedios en sensores de telemetria se ven fuertemente sesgados por lecturas anomalas o paradas prolongadas. |
| `speed_all_counter` y `trip_distance_all_counter` | Numerico | Son contadores acumulativos de hardware que dependen exclusivamente del tiempo que el sensor ha estado conectado, sin aportar una pauta de habitos de conduccion. |

---

## 3. Diseno de Features Derivadas (Feature Engineering)

Para enriquecer la informacion disponible sobre el ciclo de vida del vehiculo, la fidelidad con la marca y el comportamiento digital, se han disenado e implementado 5 nuevas variables calculadas:

### 3.1 `vehicle_age_years` (Antiguedad del vehiculo en anos)
- **Formula:** `(Fecha_Referencia - date_of_first_registration) / 365.25`
- **Justificacion:** La fecha de primera matriculacion en formato de cadena no puede procesarse directamente por un algoritmo de clustering. Al transformarla en una variable continua en anos respecto a la fecha actual del proyecto, capturamos directamente en que punto de la vida util se encuentra el automovil. Esto permite diferenciar clientes con coches a estrenar (0-2 anos), en periodo de garantia (2-4 anos) o en fase de renovacion (>5 anos).

### 3.2 `estimated_itv_passes` (Inspecciones tecnicas obligatorias estimadas)
- **Formula:** 
  - Si `vehicle_age_years < 4` -> `0`
  - Si `vehicle_age_years >= 4` -> `1 + int((vehicle_age_years - 4) / 2)`
- **Justificacion:** Basado en la normativa europea de inspeccion tecnica de vehiculos (primera revision a los 4 anos de antiguedad y posteriormente cada 2 anos). En lugar de ser solo un valor temporal lineal, modela el numero de revisiones regulatorias que el vehiculo ha debido afrontar. Es un proxy util para medir el desgaste mecanico acumulado y el coste de mantenimiento para el cliente.

### 3.3 `workshop_frequency_yearly` (Tasa anualizada de visitas al taller)
- **Formula:** `workshop_visits_warranty_goodwill / ((months_as_mb_customer / 12) + 0.1)`
- **Justificacion:** La columna bruta `workshop_visits_warranty_goodwill` acumula el total historico de visitas. Esto penalizaba a los clientes veteranos (que naturalmente tienen mas visitas) y favorecia a los clientes recien incorporados. Al dividir el numero de visitas entre los anos de relacion con Mercedes-Benz, obtenemos una tasa de intensidad de uso del servicio postventa anualizada y comparable entre cualquier tipo de cliente.

### 3.4 `marketing_receptivity_score` (Indice de receptividad comercial)
- **Formula:** `consent_phone + consent_epost + consent_post` (Suma binaria de 0 a 3)
- **Justificacion:** Sintetiza los tres campos booleanos de consentimiento de comunicacion en una escala ordinal de disposicion al contacto. Un valor 0 representa un cliente con rechazo total a promociones, mientras que un valor 3 indica que ha autorizado todos los canales (telefono, correo electronico y correo postal). Permite al modelo aislar perfiles comerciales altamente contactables.

### 3.5 `has_telemetry_active` (Indicador de conexion digital)
- **Formula:** `1 si trip_distance_median no es nulo, 0 si es nulo`
- **Justificacion:** El 60% de los registros no contienen datos de telemetria. En lugar de interpretar esto como un fallo de recopilacion de datos, refleja una realidad comercial: un 40% de los clientes utiliza activamente los servicios conectados de la marca (Mercedes me / app de telemetria), mientras que el 60% restante no los tiene activados. Convertir esta ausencia en una variable indicadora (flag) preserva ese patron digital clave para el clustering.

---

## 4. Tratamiento de Variables Categoricas y Estrategia Arquitectonica

### 4.1 Problema del One-Hot Encoding masivo
En el dataset existen variables categoricas con una cardinalidad muy elevada:
- `vehicle_type`: 1.436 categorias distintas.
- `model_series`: 88 series de modelos.
- `model_class`: 42 clases comerciales.

Aplicar One-Hot Encoding tradicional crearia una matriz dispersa con mas de 1.500 columnas binarias compuestas en su inmensa mayoria por ceros. En espacios con tantas dimensiones, la distancia euclidiana entre cualquier par de puntos tiende a igualarse (maldicion de la dimensionalidad), haciendo que K-Means agrupe de forma erratica e ineficiente.

### 4.2 Solucion: Macro-Segmentacion y Metadatos de Negocio
Para resolver esto de forma limpia y mantenible:
1. **Codificacion Ordinal de `age_range`:** El rango de edad tiene un orden natural claro (`<25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`). Se codifica como un entero de 0 a 5 mediante `OrdinalEncoder`, capturando la jerarquia de edad en una sola columna numerica sin anadir dimensiones.
2. **Particion en 4 Macro-Segmentos:** Se divide el dataset en 4 subconjuntos naturales combinando mercado (`country_code`: Suiza [CH] y Austria [AT]) y tipologia de cliente (`role_type`: Particulares [B2C] y Empresas/Flotas [B2B]). De esta forma, cada segmento responde a una dinamica de compra diferente:
   - `CH_B2C` (189.879 registros): Particulares en Suiza.
   - `AT_B2C` (107.450 registros): Particulares en Austria.
   - `CH_B2B` (2.999 registros): Empresas y flotas en Suiza.
   - `AT_B2B` (2.522 registros): Empresas y flotas en Austria.
3. **Desacoplamiento de Metadatos:** Las columnas cualitativas y descriptivas (`client_id`, `vehicle_id`, `model_series`, `model_class`, `engine_concept`, etc.) se extraen a un archivo `{segmento}_metadata.csv` que comparte el mismo identificador de fila que las matrices numericas. Esto permite que el modelo de clustering trabaje de forma rapida y limpia solo con numeros, y luego podamos cruzar las etiquetas de cluster con los metadatos para saber que vehiculos conducen y que motorizacion prefieren.

---

## 5. Tratamiento de Variables Numericas y Valores Nulos

### 5.1 Imputacion Controlada de Telemetria
Las variables continuas de conduccion (`trip_distance_median`, `speed_median`, `fuel_consumption_average`, `operating_hours_total`) presentan valores ausentes en los clientes sin servicio digital conectado.
- Se calculan las medianas de cada variable agrupando por `role_type`, `country_code` y `model_class`.
- Se imputan dichos valores neutros en los huecos vacios, y se cuenta con el respaldo de la mediana global ante cualquier grupo minoritario sin muestras.
- Como disponemos de la feature binaria `has_telemetry_active`, el algoritmo no confunde un valor imputado con una medicion real de sensor.

### 5.2 Justificacion de `RobustScaler` frente a `StandardScaler`
Para normalizar los datos se evaluan dos alternativas principales:
- `StandardScaler`: Centra la media en 0 y escala a varianza unitaria usando la media y la desviacion estandar.
- `RobustScaler`: Centra utilizando la mediana y escala utilizando el rango intercuartilico (IQR = Percentil 75 - Percentil 25).

**Por que se ha elegido `RobustScaler`:**  
Variables como `operating_hours_total` (con clientes que registran hasta 26.800 horas frente a una mediana de unas 800 horas) o `population` contienen distribuciones con asimetria positiva y colas muy largas (outliers severos).  
Si usasemos `StandardScaler`, la media y la desviacion estandar se verian arrastradas por estos pocos vehiculos extremos, comprimiendo artificialmente a casi todo el 95% de la poblacion en un rango diminuto de valores muy cercanos entre si. `RobustScaler` evita este problema por completo, ya que la mediana y el IQR no se alteran por la existencia de valores extremos, logrando un escalado mucho mas representativo para el calculo de distancias.

---

## 6. Listado Final de Features Seleccionadas

El modelo de clustering recibe un espacio de 16 variables puramente numericas y normalizadas:

| N.º | Variable | Naturaleza | Origen | Justificacion funcional |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `age_range_encoded` | Ordinal (0 a 5) | Codificacion | Posicion del cliente en su ciclo vital y capacidad de ahorro. |
| 2 | `vehicle_age_years` | Continua | Feature derivada | Antiguedad real del automovil y fase de renovacion. |
| 3 | `estimated_itv_passes` | Discreta | Feature derivada | Ciclos de inspeccion tecnica y madurez mecanica del coche. |
| 4 | `is_current_vehicle` | Binaria (0 o 1) | Limpieza / Casteo | Indica si el cliente conserva el coche o es un registro historico. |
| 5 | `months_as_mb_customer` | Continua | Original | Antiguedad de la relacion comercial con la marca. |
| 6 | `currently_connected_vehicles` | Discreta | Original | Tamano del parque movil activo asociado al titular. |
| 7 | `workshop_frequency_yearly` | Continua | Feature derivada | Frecuencia de mantenimiento oficial anualizada (fidelidad postventa). |
| 8 | `workshop_visits_warranty_goodwill` | Discreta | Original | Volumen historico de intervenciones bajo garantia o cortesia. |
| 9 | `marketing_receptivity_score` | Ordinal (0 a 3) | Feature derivada | Permisividad y canales autorizados para campanas de marketing. |
| 10 | `purchasing_power_per_inhabitant_eur`| Continua | Original | Capacidad economica media del entorno residencial del cliente. |
| 11 | `population` | Continua | Original | Entorno geografico: densidad urbana frente a zonas rurales. |
| 12 | `has_telemetry_active` | Binaria (0 o 1) | Feature derivada | Nivel de adopcion tecnologica y conectividad del usuario. |
| 13 | `trip_distance_median` | Continua | Imputada | Patron de trayectos: desplazamientos cortos urbanos vs largo recorrido. |
| 14 | `speed_median` | Continua | Imputada | Estilo de conduccion y tipo de vias habituales (ciudad vs autopista). |
| 15 | `fuel_consumption_average` | Continua | Imputada | Eficiencia energetica y tipologia de conduccion / motorizacion. |
| 16 | `operating_hours_total` | Continua | Imputada | Intensidad global de uso del vehiculo a lo largo de su vida. |

---

## 7. Construccion del Pipeline de Scikit-Learn

Para cumplir con las buenas practicas de ingenieria del software, el escalado y preparacion no se realiza mediante operaciones sueltas e irreversibles sobre los DataFrames, sino a traves de la clase `Pipeline` de Scikit-Learn:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def create_preprocessing_pipeline() -> Pipeline:
    """
    Construye el pipeline de transformacion modular.
    Aplica RobustScaler sobre las variables numericas garantizando
    que el escalador se ajuste independientemente por segmento.
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])
    return pipeline
```

### Ventajas de este diseno:
1. **Modularidad:** Si en el futuro se desea anadir una reduccion de dimensionalidad (como PCA) o un imputador adicional dentro del pipeline, puede integrarse como un nuevo paso sin reescribir el codigo de particion.
2. **Prevencion de Fuga de Datos (Data Leakage):** El pipeline se ajusta (`fit`) de forma independiente en cada macrosegmento (`CH_B2C`, `AT_B2C`, etc.), evitando que la mediana o cuartiles de un pais rico como Suiza contaminen las escalas de Austria, o que las flotas comerciales desfiguren a los particulares.
3. **Mantenibilidad:** El objeto pipeline resultante puede exportarse con `joblib` o `pickle` y utilizarse en produccion para transformar nuevos datos entrantes de un solo paso (`pipeline.transform(nuevos_datos)`).

---

## 8. Artefactos Generados y Estructura en Disco

La ejecucion del pipeline (`scripts/Reto_2_Feature_Engineering.py` y `notebooks/Reto_2_Feature_Engineering.ipynb`) genera los siguientes archivos en la carpeta `data/`:

- **Archivos para el clustering (`_features_scaled.csv`):**
  - `CH_B2C_features_scaled.csv` (189.879 filas x 16 columnas)
  - `AT_B2C_features_scaled.csv` (107.450 filas x 16 columnas)
  - `CH_B2B_features_scaled.csv` (2.999 filas x 16 columnas)
  - `AT_B2B_features_scaled.csv` (2.522 filas x 16 columnas)
  *Son los archivos directos que debe consumir el modelo K-Means.*

- **Archivos para interpretacion de medias (`_features_raw.csv`):**
  - Mismas dimensiones, con los valores en unidades reales (anos, km/h, euros, etc.) para que marketing pueda interpretar que significan los clusters obtenidos.

- **Archivos de metadatos cualitativos (`_metadata.csv`):**
  - Mismo orden de filas con `client_id`, `vehicle_id`, `model_series`, `model_class`, etc., listos para hacer un `join` tras el agrupamiento.

---

## 9. Conclusiones

1. Se ha pasado de una tabla heterogenea de 39 columnas con millones de nulos a un conjunto compacto de **16 variables normalizadas y con significado de negocio**, sin ningun valor nulo residual.
2. Se han incorporado **5 features derivadas** que aportan informacion de alto impacto sobre antiguedad, revisiones tecnicas, tasa de visitas al taller, canalidad comercial y digitalizacion.
3. Se ha evitado la dispersion masiva de columnas al sustituir el One-Hot Encoding por una **macro-segmentacion por pais y rol**, desacoplando los metadatos cualitativos.
4. El uso de **`RobustScaler`** dentro de un **`Pipeline` de Scikit-Learn** asegura que los valores atipicos en kilometrajes y horas de uso no distorsionen las agrupaciones matematicas del modelo.
5. El entorno queda completamente preparado para afrontar la fase de clustering y seleccion del K optimo.
