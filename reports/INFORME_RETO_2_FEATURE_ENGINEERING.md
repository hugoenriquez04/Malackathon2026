# Informe Tecnico — Reto 2: Feature Engineering y Preparacion de Datos para Clustering

**Proyecto:** Mercedes-Benz Hackathon — "Clustering the Star"  
**Autor:** Estudiante de Grado en Ingenieria Informatica  
**Modulo:** Reto 2 — Seleccion de variables, ingenieria de caracteristicas y pipeline de preprocesamiento  
**Notebook entregable:** `notebooks/Reto_2_Feature_Engineering.ipynb` (ejecutado y documentado)  
**Dataset de partida:** `data/reto1_dataset_limpio.csv` (302.850 filas x 26 columnas, 0 nulos)  
**Fecha:** Octubre 2026  

---

## 1. Introduccion y Objetivos de Negocio

El objetivo de este reto consiste en seleccionar, justificar, transformar y preparar las caracteristicas (features) que alimentaran el modelo de aprendizaje no supervisado (clustering).

En proyectos de mineria de datos y segmentacion de clientes, alimentar un algoritmo de agrupamiento con todas las columnas disponibles de forma indiscriminada conduce inevitablemente a malos resultados debido a tres factores criticos:
1. **Ruido e irrelevancia:** Columnas administrativas (identificadores de cliente, chasis o indices) que no describen patrones de consumo ni habitos de uso.
2. **Multicolinealidad:** Variables redundantes con correlaciones cercanas a 1.0 que sobreponderan artificialmente ciertas dimensiones en el calculo de distancias.
3. **Maldicion de la dimensionalidad:** La inclusion descontrolada de categorias nominales (mediante One-Hot Encoding) genera matrices dispersas donde las distancias euclidianas pierden su capacidad de discriminacion geometrica.

Por tanto, este trabajo aborda la preparacion de los datos pensando como un **analista de negocio de automocion**:
- ¿Que informacion es verdaderamente util para segmentar clientes en Mercedes-Benz? Ciclo de vida del automovil, fidelidad y lealtad al servicio postventa, capacidad economica del entorno, habitos de conduccion y adopcion de servicios digitales.
- ¿Como estructurar la solucion de forma reproducible? Mediante un pipeline desacoplado que recoge la salida limpia del Reto 1, crea variables sinteticas de alto valor, aisla los metadatos descriptivos y estandariza las magnitudes mediante tecnicas robustas frente a valores atipicos.

---

## 2. Seleccion y Descarte Justificado de Variables

A partir del dataset consolidado, se efectua un filtrado riguroso eliminando variables superfluas o colineales antes de conformar el espacio vectorial de agrupamiento:

| Columna | Tipo original | Motivo de exclusion |
| :--- | :--- | :--- |
| `Unnamed: 0` | Numerico | Indice autogenerado sin ningun valor analitico ni descriptivo. |
| `client_id`, `vehicle_id` | Identificadores | Granularidad a nivel de registro individual. No generalizan comportamiento. |
| `zipcode` | Categorico / Texto | Excesiva cardinalidad y granularidad dispersa. El entorno geografico ya queda capturado de forma sintetica por `country_code` y `population`. |
| `number_of_households` | Numerico | Correlacion lineal perfecta (r = 1.00) con `population`. Mantener ambas duplicaria artificialmente el peso del tamano demografico municipal. |
| `purchasing_power_per_household_eur` | Numerico | Correlacion superior a 0.96 con `purchasing_power_per_inhabitant_eur`. Se descarta para evitar colinealidad de escala socioeconomica. |
| `purchasing_power_millions_eur` | Numerico | Magnitud agregada municipal con colinealidad r > 0.96 respecto al poder adquisitivo por habitante. |
| `fuel_type` | Categorico | Presentaba un 59% de valores ausentes en origen y duplica la informacion estructurada de `engine_concept`. |
| `speed_mode`, `trip_distance_mode` | Texto | Registros con cadenas no uniformes y valores modales no continuos. |
| `speed_average`, `trip_distance_average` | Numerico | Los promedios en sensores de telemetria sufren fuerte distorsion por lecturas anomalas o paradas prolongadas; se sustituyen por las medianas (`speed_median` y `trip_distance_median`). |
| `speed_all_counter`, `trip_distance_all_counter` | Numerico | Contadores acumulativos de hardware dependientes del tiempo de conexion del sensor, sin aportar informacion sobre patrones de habitos de conduccion. |

---

## 3. Descripcion y Justificacion de las Features Derivadas Creadas

Para enriquecer el conocimiento sobre el ciclo de vida del vehiculo, la fidelidad con la marca y el comportamiento digital, se han disenado e implementado **4 nuevas caracteristicas calculadas**:

### 3.1 `vehicle_age_years` (Antiguedad del vehiculo en anos)
- **Construccion:** `(REFERENCE_DATE - date_of_first_registration) / 365.25`, acotado a `>= 0`.
- **Justificacion de negocio:** La fecha de primera matriculacion como timestamp no es interpretable por un algoritmo euclideo. Convertirla en una variable continua en anos respecto a la fecha de referencia del proyecto permite al modelo discriminar con precision clientes con vehiculos en garantia de fabrica (0 a 4 anos) de clientes con vehiculos maduros en fase de renovacion (>5 anos).

### 3.2 `workshop_frequency_yearly` (Tasa anualizada de visitas al taller)
- **Construccion:** `workshop_visits_warranty_goodwill / ((months_as_mb_customer / 12) + 0.1)`.
- **Justificacion de negocio:** El conteo absoluto de visitas al taller penalizaba a los clientes leales con muchos anos de relacion (mas visitas acumuladas) frente a clientes nuevos. Dividir las intervenciones entre los anos de antiguedad del cliente (+0.1 de seguridad para evitar indeterminaciones por division entre cero) genera una tasa anual de lealtad y uso del taller oficial comparable entre clientes noveles y veteranos.

### 3.3 `marketing_receptivity_score` (Indice de receptividad comercial)
- **Construccion:** Suma binaria (0 a 3) de `consent_phone_communication_channel`, `consent_epost_communication_channel` y `consent_post_communication_channel`.
- **Justificacion de negocio:** Sintetiza los consentimientos de privacidad en una escala ordinal del nivel de accesibilidad del cliente. Permite que los clusters sean comercialmente accionables: un cluster con score 3 admite contacto omnicanal inmediato, mientras que un cluster con score 0 requiere campanas no invasivas o canales indirectos.

### 3.4 `has_telemetry_active` (Indicador de conexion digital)
- **Construccion:** Variable binaria (1 si el cliente dispone de registros telematicos no nulos en distancia/velocidad, 0 en caso contrario).
- **Justificacion de negocio:** Cerca del 60% de los vehiculos del parque no tienen telemetria activa (carecen de hardware conectado o no han activado la app Mercedes me). Esta bandera evita que el algoritmo interprete la ausencia de datos como "kilometraje cero", diferenciando el parque de conductores digitales del parque tradicional.

*(Nota tecnica sobre ITVs estimadas: Tras analizar la redundancia funcional entre las inspecciones tecnicas y la antiguedad lineal en anos, se decidio eliminar `estimated_itv_passes` para evitar multicolinealidad en el espacio de clustering).*

---

## 4. Tratamiento de Variables Categoricas y Estrategia de Metadatos

### 4.1 Riesgo de la codificacion One-Hot masiva
El dataset cuenta con atributos cualitativos de muy alta cardinalidad:
- `vehicle_type`: 1.436 clases.
- `model_series`: 88 clases.
- `model_class`: 42 clases.

Aplicar One-Hot Encoding convencional generaria una matriz dispersa con mas de 1.500 columnas binarias dominadas por ceros, destruyendo la compacidad de la distancia euclidea.

### 4.2 Solucion Implementada: Codificacion Ordinal, Macro-Segmentacion y Metadatos
1. **Codificacion Ordinal de `age_range`:** El rango de edad presenta un orden secuencial natural (`<25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`). Se codifica de 0 a 5 mediante `OrdinalEncoder`, conservando la jerarquia vital en una sola dimension compacta.
2. **Particion en 4 Macro-Segmentos:** Se divide el problema de forma natural cruzando mercado geografico (`country_code`: Suiza [CH] y Austria [AT]) y tipologia juridica (`role_type`: Particulares [B2C] y Flotas/Empresas [B2B]). Cada segmento responde a realidades impositivas, presupuestarias y de uso totalmente distintas:
   - `CH_B2C` (189.508 clientes): Particulares Suiza.
   - `AT_B2C` (107.545 clientes): Particulares Austria.
   - `CH_B2B` (3.268 clientes): Empresas Suiza.
   - `AT_B2B` (2.529 clientes): Empresas Austria.
3. **Desacoplamiento de Metadatos Descriptivos:** Columnas cualitativas como `model_series`, `model_class`, `vehicle_type`, `engine_concept` o `finance_leasing_status` se extraen a un archivo complementario `{segmento}_metadata.csv` alineado por indice de fila. El modelo entrena de forma limpia solo sobre variables continuas/ordinales, y posteriormente se realiza un join por indice para perfilar y bautizar los clusters resultantes.

---

## 5. Tratamiento de Variables Numericas y Pipeline de Escalado

### 5.1 Integridad y Verificacion de Valores Ausentes
Gracias a la conexion con el Reto 1, el dataset de partida cuenta con 0 valores nulos. Como salvaguarda modular, el script implementa imputacion jerarquica por medianas de grupo (`role_type`, `country_code`, `model_class`) garantizando completitud absoluta (0 nulos residuales).

### 5.2 Justificacion Tecnica de `RobustScaler` frente a `StandardScaler`
En variables de comportamiento de automocion como `operating_hours_total` (con clientes comerciales que alcanzan 26.800 horas frente a medianas de 800) o `trip_distance_median`, las distribuciones presentan una asimetria positiva extrema y colas pesadas.
- **Limitacion de `StandardScaler`:** Al emplear la media y la desviacion estandar, se ve fuertemente sesgado por los valores atipicos, contrayendo artificialmente al 95% de los clientes habituales en un micro-intervalo casi plano.
- **Ventaja de `RobustScaler`:** Utiliza la **mediana y el rango intercuartilico ($IQR = Q_3 - Q_1$)**. Es matematicamente insensible a los outliers, logrando que el centroide del 50% central de la poblacion quede distribuido de manera uniforme en torno a cero sin comprimir la varianza.

### 5.3 Encapsulacion en Pipeline de Scikit-Learn
El escalado se encapsula mediante la clase `Pipeline` de Scikit-Learn:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def create_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', RobustScaler())
    ])
```
Esto garantiza modularidad, portabilidad para entornos productivos y previene la contaminacion de datos (*data leakage*) al ajustar (`fit_transform`) el transformador de manera independiente para cada macro-segmento.

---

## 6. Tabla Maestra de las 15 Features para Clustering

A continuacion se presenta la matriz completa de las 15 variables seleccionadas para alimentar los algoritmos de clustering:

| N.º | Variable | Tipo de Variable | Construccion / Origen | Aporte a la Clasificacion y Segmentacion |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `age_range_encoded` | Ordinal (0 a 5) | Codificacion ordinal de `age_range` | Captura el ciclo vital, madurez profesional y capacidad economica acumulada del cliente. |
| 2 | `vehicle_age_years` | Continua (anos) | `(ref_date - first_reg) / 365.25` | Diferencia vehiculos en garantia oficial de aquellos en ventana de renovacion o compra de nuevo modelo. |
| 3 | `is_current_vehicle` | Binaria (0 o 1) | Casteo booleano a entero | Discrimina entre el automovil activo principal y registros de vehiculos historicos o transferidos. |
| 4 | `months_as_mb_customer` | Continua (meses) | Variable original limpia | Mide la antiguedad y lealtad historica de la relacion comercial con la marca Mercedes-Benz. |
| 5 | `currently_connected_vehicles` | Discreta (conteo) | Variable original limpia | Identifica clientes multi-vehiculo y pequenos titulares de flotas familiares o profesionales. |
| 6 | `workshop_frequency_yearly` | Continua (tasa) | `visits / ((months / 12) + 0.1)` | Frecuencia anualizada de mantenimiento oficial; diferencia clientes leales a la red de desertores posventa. |
| 7 | `marketing_receptivity_score` | Ordinal (0 a 3) | Suma de 3 consentimientos | Grado de apertura comercial; permite orientar campanas directas por CRM solo a clientes receptivos. |
| 8 | `purchasing_power_per_inhabitant_eur` | Continua (euros) | Variable original limpia | Poder adquisitivo medio del area de residencia; proxy clave para capacidad de pago y gamas premium. |
| 9 | `population` | Continua (hab.) | Variable original limpia | Contexto geografico; diferencia entornos de alta densidad urbana de areas suburbanas o rurales. |
| 10 | `has_telemetry_active` | Binaria (0 o 1) | Bandera por no nulidad telem. | Distingue al usuario tecnologico conectado a la app del cliente tradicional sin servicios activos. |
| 11 | `trip_distance_median` | Continua (km) | Mediana de trayectos | Habito de desplazamiento habitual: trayectos cortos de ciudad frente a trayectos largos interurbanos. |
| 12 | `speed_median` | Continua (km/h) | Mediana de velocidad | Estilo de conduccion y via principal habitual (trafico urbano lento vs conduccion fluida en autopista). |
| 13 | `fuel_consumption_average` | Continua (l/100km o kWh) | Promedio de consumo | Huella de eficiencia y dinamica de uso segun el tamano del vehiculo y agresividad en la conduccion. |
| 14 | `operating_hours_total` | Continua (horas) | Horas acumuladas motor | Intensidad global de rodaje y desgaste acumulado a lo largo de toda la vida del automovil. |

*(Nota de depuracion por multicolinealidad: `workshop_visits_warranty_goodwill` se descarto tras la matriz de correlacion al presentar r = 0.752 con `workshop_frequency_yearly`. Se priorizo la tasa anualizada por estar normalizada en el tiempo. Tras este descarte, la correlacion maxima en todo el dataset se redujo a r = 0.293).*

---

## 7. Artefactos Entregables Generados

La ejecucion completa del cuaderno y script genera la siguiente infraestructura de datos en `data/`:
- **Archivos de clustering escalados (`_features_scaled.csv`):**
  - `CH_B2C_features_scaled.csv` (189.508 filas x 14 columnas)
  - `AT_B2C_features_scaled.csv` (107.545 filas x 14 columnas)
  - `CH_B2B_features_scaled.csv` (3.268 filas x 14 columnas)
  - `AT_B2B_features_scaled.csv` (2.529 filas x 14 columnas)
- **Archivos en escala real (`_features_raw.csv`):** Mismas 14 variables en magnitudes originales (anos, km/h, euros) para la interpretacion de centroides.
- **Archivos de metadatos (`_metadata.csv`):** Columnas cualitativas (`model_series`, `model_class`, `engine_concept`, `finance_leasing_status`, etc.) alineadas exactamente por indice de fila para la caracterizacion posterior de los clústeres.
