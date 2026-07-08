# NHANES 2021-2023 Data Processing Pipeline

This repository contains the data engineering pipeline and statistical cleaning protocols applied to the NHANES 2021-2023 dataset. The objective is to construct a rigorous, optimized feature space for linear classifiers (e.g., Logistic Regression and Support Vector Machines) while preserving structural data integrity.

## Methodology: Data Engineering and Statistical Transformations

This section details the cleaning protocols and statistical rationale applied to the NHANES 2021-2023 data modules. All transformations and inter-dataframe merging operations are strictly anchored to the respondent's unique sequence number (`SEQN`).

---

### 1. Alcohol Use (ALQ)
* **Target Feature:** `Ever have 4+ drinks every day`
* **Statistical Rationale:** The NHANES `ALQ` module relies on conditional skip patterns, resulting in structural missingness. To preserve statistical power, `ALQ151` was isolated as a robust proxy for heavy alcohol consumption history. The raw encoding was remapped to a boolean format (`0` = No, `1` = Yes), discarding uninformative classifications.

### 2. Body Measures (BMX)
* **Target Features:** `Body Mass Index`, `Waist-to-Hip Ratio`
* **Statistical Rationale:** To capture centralized adiposity without violating feature independence, a composite indicator—`Waist-to-Hip Ratio`—was engineered by dividing `BMXWAIST` by `BMXHIP`. To mitigate Variance Inflation Factors (VIF), the highly collinear primitive variables were excised from the feature space. `BMXBMI` was retained as `Body Mass Index`.

### 3. Blood Pressure Questionnaire (BPQ)
* **Target Features:** `Have Hypertension`, `High Cholesterol WITHOUT Medicine`, `High Cholesterol WITH Medicine`
* **Statistical Rationale:** Raw variables assessing chronic cardiovascular conditions (`BPQ080`, `BPQ101D`) exhibit severe multicollinearity. Using `np.where`, raw variables were synthesized into bifurcated binary features: `High Cholesterol WITHOUT Medicine` and `High Cholesterol WITH Medicine`. Because `np.where` implicitly defaults missing values to `0` (False), a subsequent logical override was enforced using `.loc[df_filtered_bpq["BPQ080"].isna()]` to inject `np.nan` back into the engineered features. This prevents structural data inflation where missing clinical records might be misinterpreted as a "negative" diagnosis. General hypertension history (`BPQ020`) was binarized to `{0, 1}`.

### 4. Blood Pressure - Oscillometric Measurement (BPXO)
* **Target Feature:** `Mean Systolic Pressure`
* **Statistical Rationale:** To mitigate measurement noise and temporary physiological spikes (e.g., "white-coat syndrome"), `Mean Systolic Pressure` was derived by calculating the row-wise arithmetic mean across the three valid oscillometric systolic measurements (`BPXOSY1`, `BPXOSY2`, `BPXOSY3`). Individual source columns were dropped to preempt structural collinearity.

### 5. Demographics (DEMO)
* **Target Features:** `Gender(Male)`, `Age`, `Ratio of Family Income to Poverty`
* **Statistical Rationale:** Demographic indicators were standardized, maintaining missing structures to preserve sample variance. The biological sex variable (`RIAGENDR`) was mapped to an asymmetric dummy indicator, `Gender(Male)`, where females correspond to `0` and males to `1`. It is important to emphasize that `Ratio of Family Income to Poverty` represents the family income divided by the United States' federal poverty threshold; values are right-censored at a maximum value of 5.0000 to protect participant confidentiality while capturing socioeconomic gradients.

### 6. Diabetes (DIQ)
* **Target Feature:** `Has Diabetes`
* **Statistical Rationale:** Survey outputs (`DIQ010`) were mapped into a definitive binary variable, `Has Diabetes`. Notably, the mapping `{2 : 0, 1 : 1, 3: 1}` was employed; by grouping borderline/pre-diabetic survey responses (code 3) into the positive target class (`1`), the pipeline ensures the model captures the full spectrum of metabolic dysregulation.

### 7. Depression Screener (DPQ)
* **Target Feature:** `Mental Health`
* **Statistical Rationale:** The NHANES protocol encodes "Refused" as `7` and "Don't Know" as `9`. Left unadjusted, these values introduce extreme statistical bias into additive metrics. These specific responses were systematically mapped to `np.nan`. A continuous aggregate severity score, `Mental Health`, was computed by summing all items across rows (`DPQ010` through `DPQ100`). Crucially, while traditional clinical scoring of the PHQ-9 often omits the final functional impact question (`DPQ100`) from the symptoms score, it was intentionally incorporated into this engineered feature. The degree to which depressive symptoms disrupt daily function provides high-value discriminatory variance for algorithmic classification. Rows containing partial missing data were evaluated using `skipna=False` to strictly enforce score integrity.

### 8. Medical Conditions (HUQ)
* **Target Feature:** `General Health Condition (Self-evaluated)`
* **Statistical Rationale:** To streamline variance from the 5-point ordinal scale (`HUQ010`), values 1, 2, and 3 were mapped to `1` (Excellent/Very Good/Good condition), while values 4 and 5 were mapped to `0` (Fair/Poor condition).

### 9. Physical Activity (PAQ)
* **Target Features:** `Total LTPA (minutes/week)`, `Sedentary activity (minutes/week)`
* **Statistical Rationale:** This module necessitated rigorous scaling alignment and a multi-tiered missing data protocol to synthesize disparate time measurements into a cohesive metabolic baseline:
  1. **Anchor Preservation:** List-wise deletion was applied strictly to the 18 observations lacking data in fundamental anchor variables (`PAD790Q` and `PAD680`). This isolated removal protects the broader structural and mathematical logic of the pipeline.
  2. **Null-Impact Imputation:** Standard structural missingness (`NaN` values resulting from valid skip patterns) was mapped to `0`. This mathematically represents an absence of reported activity, accurately reflecting a null impact on total metabolic expenditure.
  3. **Outlier Isolation:** NHANES-specific missingness codes for "Refused" (`7777`) and "Don't Know" (`9999`) were explicitly isolated and converted back to `np.nan` prior to arithmetic operations, insulating the aggregate summations from severe numerical bias.
  
  Following data cleaning, byte-encoded frequency units (`b'D'`, `b'W'`, `b'M'`, `b'Y'`) were standardized using a specialized mapping dictionary (`remapping_unity`) to yield a uniform weekly multiplier. Time allocations were then converted into minutes per week. Following epidemiological weighting paradigms established by the World Health Organization (WHO)—which state that one minute of vigorous-intensity physical activity delivers metabolic benefits equivalent to two minutes of moderate-intensity activity—vigorous-intensity minutes were mathematically doubled and summed with moderate minutes to construct the unified `Total LTPA (minutes/week)`. Sedentary duration (`PAD680`) was multiplied by 7 to define `Sedentary activity (minutes/week)`.

### 10. Sleep Disorders (SLQ)
* **Target Feature:** `Mean Sleep Time`
* **Statistical Rationale:** To create a consolidated physiological baseline, a row-wise arithmetic mean was calculated across workday (`SLD012`) and weekend (`SLD013`) sleep durations. Primitive variables were removed to prevent collinearity.

### 11. Weight History (WHQ)
* **Target Features:** `Weight Variation Within 1 Year`, `Tried to Lose Weight Within 1 Year`
* **Statistical Rationale:** Prior to calculating the weight delta, survey responses for `7777` ("Refused") and `9999` ("Don't Know") were systematically replaced with `np.nan` to prevent severe numerical bias from contaminating downstream operations. An explicit index-matching operation via `SEQN` was executed to merge the clinically measured baseline body weight from the examination module (`BMXWT`) with the self-reported historical questionnaire data (`WHD050`). Because `BMXWT` is recorded using the international metric system (**Kilograms**) whereas `WHD050` is reported by participants in the US imperial system (**Pounds**), the historical metric was divided by the conversion constant `2.20462` to normalize the scales. The continuous biomarker `Weight Variation Within 1 Year` was subsequently derived as the direct physiological delta between the examiner-measured weight and the participant's recalled weight from one year prior to the interview. Concurrently, the behavioral weight loss attempt parameter (`WHQ070`) was standardized to a clean binary dummy variable (`Tried to Lose Weight Within 1 Year`).

---