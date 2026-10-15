# 📊 INFORME EJECUTIVO & TÉCNICO — RETOS 1 Y 2
## Mercedes-Benz Hackathon: "Clustering the Star"
### Diagnóstico de Datos, EDA, Limpieza Avanzada e Ingeniería de Características

---

## 1. Resumen Ejecutivo

Este documento detalla la metodología y los resultados del preprocesamiento, análisis exploratorio (EDA) y feature engineering aplicados sobre el dataset de **302.850 clientes** de Mercedes-Benz en **Suiza (CH)** y **Austria (AT)**, abarcando clientes particulares (**B2C**) y corporativos (**B2B**).

### Principales Logros:
- **0 nulos residuales**: Se resolvieron los **2.273.248 valores nulos** iniciales sin pérdida de registros.
- **Multicolinealidad resuelta**: Detección y eliminación de redundancias perfectas ($r=1.00$) y cuasi-perfectas ($r>0.96$).
- **Imputación vectorizada de `engine_concept`**: Rescate de **36.614 valores (27.2%)** mediante inferencia jerárquica por catálogo MB.
- **Creación de 5 KPIs estratégicos** de negocio de alto impacto.
- **Particionado en 4 macro-segmentos** (`CH_B2C`, `CH_B2B`, `AT_B2C`, `AT_B2B`) con **16 variables normalizadas** mediante `RobustScaler` preparadas para clustering.

---

## 2. Diagnóstico Inicial del Dataset

| Métrica | Valor |
|---------|-------|
| Registros iniciales | **302.850** |
| Columnas iniciales | **39** |
| Total celdas nulas | **2.273.248 (19.2% del dataset)** |
| Distribución por País | Suiza (CH): 192.878 (63.7%) \| Austria (AT): 109.972 (36.3%) |
| Distribución por Canal | B2C: 297.329 (98.2%) \| B2B: 5.521 (1.8%) |

### Bloques Críticos de Nulidad:
1. **Telemetría conectada (~60% nulos):**
   - `trip_distance_average`: 61.3% nulos
   - `speed_average`: 60.2% nulos
   - `fuel_consumption_average`: 59.6% nulos
   - `operating_hours_total`: 57.5% nulos
   - *Diagnóstico*: No es un fallo de recogida, sino un indicador de **penetración digital del cliente**.

2. **Concepto de Motor (`engine_concept`):** 134.423 nulos (44.4%).
3. **Poder Adquisitivo Socioeconómico:** 19.011 nulos (6.28%) en variables socio-demográficas por código postal.

---

## 3. Decisiones Arquitectónicas y de Limpieza (Reto 1)

### 3.1. Tratamiento de Multicolinealidad
- **Demografía**: `population` y `number_of_households` presentaban una correlación de $r = 1.00$. Se eliminó `number_of_households` por redundancia estricta.
- **Poder adquisitivo**: `purchasing_power_per_inhabitant_eur`, `per_household_eur` y `millions_eur` tenían $r > 0.96$. Se conservó únicamente `purchasing_power_per_inhabitant_eur` normalizada.
- **Telemetría paramétrica vs no paramétrica**: `speed_average` y `speed_median` ($r = 0.88$). Se conservó la mediana por su resistencia a valores atípicos de autopista.

### 3.2. Imputación Jerárquica de `engine_concept`
En lugar de sustituir 134K valores por "Unknown" o la moda global (lo que sesgaría hacia diésel o gasolina):
1. Se explotó la jerarquía de catálogo de Mercedes-Benz: `model_series` + `vehicle_type` $\rightarrow$ `engine_concept`.
2. Se calculó la moda condicionada vectorizada mediante merge:
   $$\text{engine\_concept}_{\text{imputed}} = \text{Mode}(\text{engine\_concept} \mid \text{model\_series}, \text{vehicle\_type})$$
3. Se recuperaron **36.614 registros** con su motorización real de catálogo; el remanente sin referencia se codificó como "Unknown".

### 3.3. Resolución del 60% de Nulos en Telemetría
La hipótesis clave: un cliente sin datos de telemetría es un cliente con vehículo desconectado o sin servicios Mercedes me activados.
- Se creó la variable indicadora:
  $$\text{has\_telemetry\_active} \in \{0, 1\}$$
- Se imputaron las variables continuas de telemetría a 0. De este modo, la distancia euclidiana en el espacio de clustering separa limpiamente la **población digitalizada** de la **analógica** sin distorsionar la varianza.

---

## 4. Feature Engineering: Nuevos KPIs de Negocio (Reto 2)

Se diseñaron 5 nuevas variables de valor estratégico para la toma de decisiones:

1. **`vehicle_age_years`**: Antigüedad real del vehículo calculada respecto a la fecha de referencia del estudio (2024.0).
2. **`estimated_itv_passes`**: Inspecciones técnicas obligatorias superadas según la legislación aplicable en CH y AT ($4-2-2$ años).
3. **`workshop_frequency_yearly`**:
   $$\text{workshop\_frequency\_yearly} = \frac{\text{workshop\_visits\_total}}{\max(\text{months\_as\_mb\_customer} / 12, 0.5)}$$
4. **`marketing_receptivity_score`**: Puntuación aditiva $[0, 5]$ de permisos comerciales (email, teléfono, SMS, app, correo postal).
5. **`is_current_vehicle`**: Flag binario que distingue si el cliente aún posee el vehículo o es un registro histórico.

---

## 5. Particionado y Escalado de Datos

Para evitar la *maldición de la dimensionalidad* producida por codificar más de 1.400 `vehicle_types` en One-Hot Encoding, se adoptó una arquitectura por particiones naturales de negocio:

| Partición | Registros | Proporción | Escalador |
|-----------|-----------|:----------:|:---------:|
| **`CH_B2C`** | 189.879 | 62.7% | RobustScaler |
| **`CH_B2B`** | 2.999 | 1.0% | RobustScaler |
| **`AT_B2C`** | 107.450 | 35.5% | RobustScaler |
| **`AT_B2B`** | 2.522 | 0.8% | RobustScaler |

### Features seleccionadas para Clustering (16 variables):
- `age_range_encoded`
- `vehicle_age_years`
- `estimated_itv_passes`
- `is_current_vehicle`
- `months_as_mb_customer`
- `currently_connected_vehicles`
- `workshop_frequency_yearly`
- `workshop_visits_warranty_goodwill`
- `marketing_receptivity_score`
- `purchasing_power_per_inhabitant_eur`
- `population`
- `has_telemetry_active`
- `trip_distance_median`
- `speed_median`
- `fuel_consumption_average`
- `operating_hours_total`

*Escalado*: Se utilizó `RobustScaler` (basado en mediana y rango intercuartílico IQR) debido a las distribuciones con asimetría positiva extrema (visitas a taller, horas operativas y población).
