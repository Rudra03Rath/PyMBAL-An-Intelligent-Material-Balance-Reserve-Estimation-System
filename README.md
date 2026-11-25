# PyMBAL — Intelligent Material Balance & Reserve Evaluation System

**PyMBAL** is a Python-based reservoir engineering application that integrates PVT preparation, material-balance analysis, coupled gas-cap/aquifer model selection, drive-mechanism interpretation, volumetric estimation, forecasting, decline-curve analysis, and technical-reserves evaluation in one workflow.

The project is designed as an engineering decision-support and learning tool. It converts reservoir production and pressure history into a set of competing physical interpretations, quantifies uncertainty/identifiability where implemented, and presents the results through a Streamlit dashboard and structured report.

> **Important:** PyMBAL is not a commercial reservoir simulator or a substitute for engineering judgement. A mathematically best-fitting model is not necessarily a uniquely proven physical description of the reservoir. Review competing candidates, identifiability warnings, input assumptions, and independent reservoir information before interpreting results.

---

## Contents

- [Capabilities](#capabilities)
- [End-to-end workflow](#end-to-end-workflow)
- [Engineering methods](#engineering-methods)
  - [PVT engine](#1-pvt-engine)
  - [Material balance and underground withdrawal](#2-material-balance-and-underground-withdrawal)
  - [Gas-cap and aquifer model selection](#3-gas-cap-and-aquifer-model-selection)
  - [Drive-mechanism interpretation](#4-drive-mechanism-interpretation)
  - [OOIP and GIIP estimation](#5-ooip-and-giip-estimation)
  - [Forecasting, DCA, and reserves](#6-forecasting-dca-and-reserves)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Run the application](#run-the-application)
- [Input data format](#input-data-format)
- [Synthetic validation datasets](#synthetic-validation-datasets)
- [Testing](#testing)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Future improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## Capabilities

| Module | Functionality |
|---|---|
| Data ingestion and QC | Loads reservoir history, validates required data, cleans and prepares the dataset |
| PVT engine | Laboratory, correlation, and hybrid PVT workflows |
| Reservoir state | Loads initial reservoir/fluid parameters and calculates initial PVT state |
| Underground withdrawal | Computes oil, gas, and water withdrawal on a consistent reservoir-volume basis |
| Material balance | Calculates expansion/withdrawal terms and performs MBE-based analysis, including Havlena–Ode estimation |
| Coupled model selection | Evaluates alternative gas-cap and aquifer configurations against the observed reservoir history |
| Aquifer modelling | Supports Schilthuis, Fetkovich, and Van Everdingen–Hurst water-influx formulations |
| Gas-cap calibration | Estimates/compares gas-cap contribution as part of coupled reservoir-model calibration |
| Drive interpretation | Evaluates contributions from solution-gas expansion, gas-cap expansion, water influx, and other expansion terms |
| Volumetrics | Produces OOIP and, for gas-cap cases, GIIP estimates for successful candidate interpretations |
| Material-balance forecasting | Projects reservoir-state/pressure behaviour under specified future assumptions |
| Decline Curve Analysis | Fits and compares production-decline models and generates rate/cumulative forecasts |
| Technical reserves | Calculates EUR and remaining technical volumes, applying an OOIP constraint where needed |
| Reporting and visualization | Provides candidate comparison tables, drive-index summaries, plots, warnings, and a structured report |
| User interface | Streamlit dashboard for CSV upload, model constraints, forecast assumptions, and result exploration |

The application deliberately preserves multiple successful model candidates and flags ambiguous interpretations instead of hiding them behind a single output.

## End-to-end workflow

```text
Reservoir History (CSV) + Reservoir Configuration
                       |
                       v
              Data Loading & QC
                       |
                       v
                  PVT Engine
          Laboratory / Correlation / Hybrid
                       |
                       v
             Initial Reservoir State
                       |
                       v
             Underground Withdrawal
                       |
                       v
          Expansion and Material Balance
                       |
                       v
     Coupled Gas-Cap / Aquifer Model Selection
          |             |              |
          |             |              |
        No influx     Schilthuis     Fetkovich / V-E-H
          \             |              /
           \____________|_____________/
                       |
                       v
          Candidate Ranking & Identifiability
                       |
             __________|__________
            |          |          |
            v          v          v
       Drive Index  OOIP/GIIP  MBE Forecast
       Analysis     Volumetrics
            |          |          |
            |__________|__________|
                       |
                       v
             DCA Forecast Scenarios
                       |
                       v
             EUR / Technical Reserves
                       |
                       v
          Visualizations and Engineering Report
```

---

## Engineering methods

### 1. PVT engine

The PVT engine supports three operating modes:

- **Laboratory mode:** uses supplied PVT measurements and pressure-based interpolation when laboratory tables are available.
- **Correlation mode:** estimates selected fluid properties from fluid characteristics and reservoir conditions.
- **Hybrid mode:** retains available measured properties, uses laboratory interpolation for missing values where possible, and calculates remaining missing properties using configured correlations.

Implemented calculation methods include:

| Fluid | Method | Main outputs |
|---|---|---|
| Oil | Standing correlations | Solution gas-oil ratio (`Rs`), bubble-point pressure (`Pb`), oil formation volume factor (`Bo`) |
| Gas | Sutton pseudo-critical properties and Hall–Yarborough | Pseudo-critical/reduced properties, gas compressibility factor (`Z`), gas formation volume factor (`Bg`) |
| Water | McCain correlations | Water formation volume factor (`Bw`) and water viscosity |

The PVT engine also contains interpolation and validation logic. Correlation-derived properties are estimates and should be checked against representative laboratory PVT data when available.

### 2. Material balance and underground withdrawal

The core workflow prepares the reservoir state, computes cumulative underground withdrawal, evaluates fluid/rock expansion terms, and applies material-balance formulations to the observed pressure/production history.

A conventional oil-reservoir MBE can be represented in the form:

[
F=N(E_o+mE_g^*+E_{fw})+W_e
]

where:

- (F) = cumulative underground withdrawal
- (N) = original oil in place
- (E_o) = oil and solution-gas expansion term
- (m) = initial gas-cap ratio
- (E_g^*) = gas-cap expansion term
- (E_{fw}) = rock and connate-water expansion term
- (W_e) = cumulative water influx

The implementation separates reservoir-state, expansion, gas-cap, material-balance, and underground-withdrawal calculations into dedicated modules. Consistent units and compatible volume bases are essential.

### 3. Gas-cap and aquifer model selection

PyMBAL evaluates alternative reservoir configurations rather than assuming one drive mechanism in advance. Model constraints can be supplied by the user (for example, gas-cap or aquifer presence, or a specified aquifer model); leaving the relevant constraints unknown allows the selection workflow to evaluate admissible candidates.

Supported water-influx model families:

| Model | Representation | Key modelling idea |
|---|---|---|
| Schilthuis | Steady-state aquifer | Water influx rate responds to the pressure difference through an influx/productivity coefficient |
| Fetkovich | Finite aquifer | Represents finite aquifer storage and pressure depletion |
| Van Everdingen–Hurst | Transient aquifer | Uses dimensionless transient influx response and pressure-history superposition |

The calibration/selection layer compares model performance and physical validity, including the associated OOIP and gas-cap estimates. The selected model should be interpreted alongside the competitive candidate table and identifiability status.

### 4. Drive-mechanism interpretation

The drive-mechanism interpretation layer consumes model-selection results and evaluates the relative contribution of the reservoir energy/expansion mechanisms represented by the selected case.

Outputs include:

- Final drive-mechanism contribution/drive indices.
- Drive-index history over the observed production period.
- Dominant and overall drive interpretation.
- Evidence and warnings associated with the selected configuration.

Drive indices are model-dependent diagnostics. They should not be treated as direct measurements, and a single production trend (such as rising GOR or water cut) is not, by itself, proof of a particular drive mechanism.

### 5. OOIP and GIIP estimation

The volumetric analysis layer converts successful model-selection candidates into OOIP and, where a gas cap is present, GIIP estimates.

It retains candidate-level estimates and competitive ranges rather than automatically averaging physically different interpretations. The selected numerical result remains conditional on the data, model family, configuration, and calibration assumptions.

### 6. Forecasting, DCA, and reserves

PyMBAL includes two distinct forecasting workflows:

**Material-balance reservoir-state forecast:** projects reservoir-state/pressure behaviour from the selected material-balance interpretation under user-specified future assumptions, such as forecast horizon, pressure-decline fraction, and expansion endpoint multiplier.

**Decline Curve Analysis (DCA):** evaluates production-rate decline models, compares successful fits, and constructs forecast rate/cumulative-production profiles. The selected DCA model is a mathematical fit ranking; it is not automatically the most geologically plausible long-term forecast.

The reserves layer uses DCA EUR and historical cumulative production, with the material-balance OOIP acting as an upper technical constraint. It reports raw versus constrained EUR and remaining technical reserves where applicable.

> **Terminology:** OOIP/GIIP are in-place quantities. EUR is a forecasted ultimate production quantity. Remaining technical reserves in this project are model-derived technical estimates; they are not independently certified commercial reserves or a formal reserves classification.

---

## Repository structure

The principal project layout is:

```text
PyMBAL/
│
├── app.py
├── requirements.txt
├── README.md
├── generate_8_coupled_synthetic_datasets.py
│
├── data/
│   ├── reservoir_config/
│   │   └── reservoir_parameters.json
│   └── sample_data/
│       ├── sample_material_balance_data.csv
│       ├── sample_correlation_pvt_data.xlsx
│       ├── sample_hybrid_pvt_data.xlsx
│       ├── synthetic_closed_oil_reservoir.csv
│       ├── synthetic_gas_cap_reservoir.csv
│       ├── synthetic_water_influx_reservoir.csv
│       ├── synthetic_fetkovich_reservoir.csv
│       └── synthetic_8_cases/
│
├── src/
│   ├── analysis/
│   │   ├── drive_mechanism.py
│   │   ├── model_selection.py
│   │   ├── reserves.py
│   │   └── volumetrics.py
│   ├── core/
│   │   ├── expansion.py
│   │   ├── gas_cap.py
│   │   ├── gas_cap_calibration.py
│   │   ├── material_balance.py
│   │   ├── reservoir.py
│   │   └── underground_withdrawal.py
│   ├── forecasting/
│   │   ├── forecasting.py
│   │   └── scenarios.py
│   ├── influx/
│   │   ├── base.py
│   │   ├── calibration.py
│   │   ├── fetkovich.py
│   │   ├── gas_cap_aquifer_calibration.py
│   │   ├── mb_calibration.py
│   │   ├── model_selection.py
│   │   ├── schilthuis.py
│   │   └── van_everdingen_hurst.py
│   ├── models/
│   │   └── synthetic_reservoir.py
│   ├── pvt/
│   │   ├── correlations.py
│   │   ├── gas.py
│   │   ├── interpolation.py
│   │   ├── oil.py
│   │   ├── pvt_engine.py
│   │   └── water.py
│   ├── reporting/
│   │   └── report.py
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── utils/
│   │   ├── cleaner.py
│   │   ├── loader.py
│   │   └── validator.py
│   └── visualization/
│       └── plots.py
│
└── tests/
    ├── test_8_coupled_scenarios.py
    ├── test_drive_mechanism.py
    └── additional module-level tests
```

---

## Installation

### Prerequisites

- Python 3.10 or later recommended.
- Git.
- A terminal or command prompt.

### 1. Clone the repository

```bash
git clone https://github.com/Rudra03Rath/PyMBAL-An-Intelligent-Material-Balance-Reserve-Estimation-System.git
cd PyMBAL-An-Intelligent-Material-Balance-Reserve-Estimation-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current `requirements.txt` includes NumPy, pandas, Matplotlib, Plotly, SciPy, Streamlit, openpyxl, scikit-learn, and ReportLab.

---

## Run the application

The primary user interface is the Streamlit dashboard.

From the project root, run:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit in your browser.

### Dashboard workflow

1. Upload a reservoir-history **CSV** file using the sidebar.
2. Select the reservoir configuration JSON file (the default configuration is under `data/reservoir_config/`).
3. Specify gas-cap presence and aquifer presence as **Unknown**, **Present**, or **Absent**, as appropriate.
4. Optionally specify a gas-cap ratio (m) or a particular aquifer model.
5. Set future forecast assumptions (forecast horizon, pressure-decline fraction, and expansion endpoint multiplier).
6. Run the pipeline and inspect the executive summary, candidate models, drive-mechanism analysis, volumetrics, forecast, DCA, reserves, and engineering report tabs.

Leaving constraints as **Unknown** allows the model-selection layer to evaluate the admissible configurations. Supplying a constraint narrows the set of candidate cases and should be based on independent geological, petrophysical, or field evidence.

### Command-line execution

The project also retains a command-line wrapper:

```bash
python app.py
```

This uses the default input/configuration paths defined in `app.py` and prints a summary. The Streamlit dashboard is the intended interactive workflow.

---

## Input data format

The dashboard accepts a reservoir-history CSV. The pipeline validates required columns before analysis. The principal historical data fields used by the project include:

| Column | Meaning | Expected unit |
|---|---|---|
| `Date` | Observation date | Date |
| `Days` | Elapsed time | days |
| `Np_STB` | Cumulative oil production | STB |
| `Gp_Mscf` | Cumulative gas production | Mscf |
| `Wp_STB` | Cumulative water production | STB |
| `Pressure_psia` | Average reservoir pressure | psia |

PVT preparation may additionally use:

| Column | Meaning | Unit |
|---|---|---|
| `Temperature_F` | Reservoir temperature | °F |
| `Oil_API` | Stock-tank oil gravity | °API |
| `Gas_Gravity` | Gas specific gravity relative to air | dimensionless |
| `Water_Salinity_wt_pct` | Water salinity | wt% |
| `Bo_rb_per_STB` | Oil formation volume factor | rb/STB |
| `Bg_rb_per_scf` | Gas formation volume factor | rb/scf |
| `Bw_rb_per_STB` | Water formation volume factor | rb/STB |
| `Rs_scf_per_STB` | Solution gas-oil ratio | scf/STB |

A separate laboratory PVT table, when used, should include `Pressure_psia` and the relevant PVT property columns.

**Unit consistency warning:** if gas production is stored in Mscf and (B_g) is in rb/scf, convert gas production to scf before multiplying by (B_g). Also ensure that pressure is average reservoir pressure appropriate to the tank model—not flowing tubing pressure or flowing bottom-hole pressure.

For the exact required columns and validation rules, refer to the data validation and pipeline input checks in the source code. The sample files under `data/sample_data/` provide starting points for the supported workflows.

---

## Synthetic validation datasets

The repository includes synthetic reservoir histories used to exercise the engineering workflow, including closed oil, gas-cap, and water-influx cases.

The script `generate_8_coupled_synthetic_datasets.py` generates eight coupled scenarios:

1. No gas cap and no aquifer.
2. Gas cap without aquifer.
3. Schilthuis aquifer without gas cap.
4. Fetkovich aquifer without gas cap.
5. Van Everdingen–Hurst aquifer without gas cap.
6. Gas cap with Schilthuis aquifer.
7. Gas cap with Fetkovich aquifer.
8. Gas cap with Van Everdingen–Hurst aquifer.

Generate the datasets from the project root:

```bash
python generate_8_coupled_synthetic_datasets.py
```

The generated CSV files are written under:

```text
data/sample_data/synthetic_8_cases/
```

These scenarios are useful for controlled testing because their generating assumptions are known. Recovering those assumptions is a test of the implementation under the synthetic model—not proof that the same model will be uniquely identifiable in a real reservoir.

---

## Testing

Run the available tests from the repository root:

```bash
python -m pytest
```

Or run a particular test module, for example:

```bash
python -m pytest tests/test_8_coupled_scenarios.py
python -m pytest tests/test_drive_mechanism.py
```

Some legacy or module-specific test scripts may also exist at the repository root. Use the tests relevant to the module being changed and add regression tests when modifying engineering equations, unit conversions, calibration bounds, or model-selection logic.

---

## Assumptions and limitations

1. **Input quality governs output quality.** Production allocation, pressure surveys, PVT quality, dates, and unit consistency directly affect the results.
2. **Average reservoir pressure matters.** Material balance requires representative average reservoir pressure for the tank under study. Flowing pressure measurements cannot be substituted without a justified conversion/model.
3. **Tank assumptions matter.** A single-tank MBE may not represent strongly compartmentalized reservoirs, major pressure gradients, or complex communication.
4. **Model non-uniqueness is real.** Different combinations of OOIP, gas-cap size, and aquifer parameters may produce similar history matches. A low error metric does not establish a unique geological explanation.
5. **Aquifer formulations have different assumptions.** Schilthuis, Fetkovich, and Van Everdingen–Hurst should not be treated as interchangeable curve-fitting options; select/calibrate them in light of aquifer geometry, finite/infinite behaviour, and pressure history.
6. **PVT correlations are empirical.** Their reliability depends on fluid composition and the range of applicability. Laboratory measurements should take precedence when representative and reliable.
7. **Forecasts are conditional.** Material-balance forecasts and DCA forecasts depend on user-selected assumptions and extrapolation of historical behaviour. Forecast uncertainty generally increases with forecast duration.
8. **DCA ranking is mathematical.** A best-fit decline model may be statistically preferred over the observed history but physically inappropriate beyond the history period.
9. **Technical reserves are not booked reserves.** Commerciality, project maturity, operating constraints, economic limits, and the applicable reserves classification framework are outside a purely technical calculation.
10. **Synthetic data are not field validation.** Synthetic cases verify controlled behaviours; representative field data and independent benchmark comparisons are still required for broader validation.

---

## Future improvements

Potential next steps include:

- Broader unit and dimensional-consistency checks across all modules.
- More benchmark problems and comparisons against independently verified calculations.
- Stronger uncertainty quantification for OOIP, GIIP, aquifer parameters, forecasts, and reserves.
- Additional fluid models for compositionally complex oils and gas condensates.
- More explicit diagnostic plots and sensitivity summaries for model identifiability.
- Downloadable, reproducible engineering reports with input assumptions and calculation provenance.
- Field-data validation and calibration workflows.

---

## Disclaimer

PyMBAL is an academic and portfolio project developed for petroleum-engineering analysis and learning. It is provided “as is”, without warranty of accuracy or fitness for a particular purpose. Its outputs must be independently reviewed and verified before being used in operational, investment, development-planning, or reserves-reporting decisions. The user is responsible for checking the input data, units, assumptions, model applicability, and applicable professional standards.
