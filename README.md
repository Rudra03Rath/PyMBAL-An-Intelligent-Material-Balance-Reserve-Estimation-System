PyMBAL: An Intelligent Material Balance & Reserve Estimation System
PyMBAL is a Python-based petroleum engineering project designed to
support reservoir-fluid property preparation and, as the project
develops, material-balance analysis, reservoir-drive interpretation,
hydrocarbon-in-place estimation, and production/reserves evaluation.

The engineering objective is to help a reservoir engineer organize
production, pressure, and PVT data into a consistent workflow for
understanding reservoir depletion and estimating reservoir performance.
Project status: The public repository currently contains the
data-loading, validation, cleaning, and PVT-preparation workflow. The
broader material-balance, aquifer-calibration, drive-mechanism,
forecasting, and reserves capabilities described below are planned
development stages and should not be interpreted as fully implemented
in the current release.

Table of contents
- Project objectives
- Current implementation
- Engineering workflow
- PVT calculation methods
- Input data
- Installation
- Running the current workflow
- Engineering background
- Planned development roadmap
- Current limitations and engineering
  cautions
- Repository structure
- Disclaimer
Project objectives
The long-term objectives of PyMBAL are to:
1. Load and validate reservoir production, pressure, and PVT datasets.
2. Prepare PVT properties at the pressure observations used in
   reservoir analysis.
3. Calculate underground withdrawal and fluid/rock expansion terms.
4. Apply suitable material-balance formulations, including Havlena--Ode
   analysis.
5. Evaluate the possible contributions of solution-gas expansion,
   gas-cap expansion, rock/connate-water expansion, and aquifer influx.
6. Estimate OOIP and GIIP where the selected model and available data
   support the calculation.
7. Compare and calibrate suitable aquifer models.
8. Support reservoir-performance interpretation, decline-curve
   forecasting, and uncertainty analysis.
Current implementation
The current public codebase includes:
  Component                           Current capability
  Data loading                        Reads CSV and Excel (.xlsx,
.xls) files

  Data validation                     Checks required columns, missing
values, negative values, and
availability of PVT properties

  Data cleaning                       Converts dates, sorts records
chronologically, removes exact
duplicate rows, and flags unusual
cumulative-production or pressure
trends

  Laboratory PVT mode                 Uses supplied PVT properties or
interpolates available laboratory
PVT data to requested reservoir
pressures

  Correlation PVT mode                Calculates selected oil, gas, and
water properties using the
correlations listed below
Hybrid PVT mode                     Preserves available properties,
uses laboratory interpolation for
missing properties when supplied,
and calculates remaining missing
properties using the selected
correlations
Implemented PVT correlations
  Fluid                   Implemented method      Properties/calculations
  Oil                     Standing                Solution GOR (Rs),
bubble-point pressure
(Pb), oil FVF (Bo)

  Gas                     Sutton pseudo-critical  Pseudo-critical and
properties +            pseudo-reduced
Hall--Yarborough        properties, gas
compressibility factor
(Z), gas FVF (Bg)
Water                   McCain                  Water FVF (Bw) and
water viscosity
The correlation registry also contains metadata for other correlations,
including Vasquez--Beggs, Glaso, and Dranchuk--Abou-Kassem. Their
presence in the registry does not mean that their calculation
routines are currently implemented or selectable through the PVT engine.
Engineering workflow
Current workflow
Reservoir production / pressure dataset
                |
                v
          Data loading
                |
                v
      Required-column validation
                |
                v
       Data quality checks
                |
                v
        Date sorting / cleaning
                |
                v
             PVT Engine
       /          |          \
 Laboratory   Correlation   Hybrid
       \          |          /
                |
                v
       Prepared PVT dataset
Intended future workflow
Prepared reservoir and PVT data
                |
                v
 Reservoir-state calculations
                |
                v
 Underground withdrawal
                |
                v
 Material Balance Equation
                |
                v
 Gas-cap / aquifer evaluation
                |
                v
 OOIP / GIIP estimation
                |
                v
 Drive-mechanism interpretation
                |
                v
 Forecasting / recovery evaluation
                |
                v
 Sensitivity and uncertainty analysis
The second diagram is the development roadmap, not a description of
functionality currently available in the public code.
PVT calculation methods
The PVT engine supports three modes.
1. Laboratory mode
Uses PVT properties supplied by the user. If a separate laboratory PVT
table is provided, the current implementation linearly interpolates
available properties against pressure to obtain values at the reservoir
pressure observations.

Extrapolation outside the laboratory pressure range is disabled by
default. Enabling extrapolation should be done cautiously and documented
because a mathematical extrapolation is not a substitute for
representative laboratory measurements.
2. Correlation mode
Calculates selected PVT properties using empirical correlations:
- Oil: Standing correlations for Rs, Pb, and Bo.
- Gas: Sutton pseudo-critical properties and Hall--Yarborough
  calculation of Z, followed by Bg.
- Water: McCain correlations for Bw and water viscosity.
Correlation accuracy depends on fluid composition, input quality, and
the range over which the correlation is appropriate. Correlation-derived
values should be treated as estimates and checked against laboratory PVT
data when available.
3. Hybrid mode
The hybrid workflow preserves PVT properties already present in the
reservoir dataset. Where properties are missing, it uses a supplied
laboratory PVT table first, then calculates remaining missing properties
using the configured correlations.

For a defensible engineering study, the source of every PVT property
should be traceable (laboratory, correlation, or other source), and
mixed-source datasets should be reviewed for consistency.
Input data
The base reservoir dataset is expected to contain the following columns:

  Column            Description                   Unit / format
  Date            Observation date              Date
Days            Elapsed time                  days
Np_STB          Cumulative oil production     STB
Gp_Mscf         Cumulative gas production     Mscf
Wp_STB          Cumulative water production   STB
Pressure_psia   Reservoir pressure            psia

For correlation-based PVT calculations, the current validation workflow
expects:

  Column                    Description                            Unit
  Temperature_F           Reservoir temperature                  °F
Oil_API                 Stock-tank oil gravity                 °API
Gas_Gravity             Gas specific gravity relative to air   dimensionless
Water_Salinity_wt_pct   Water salinity                         wt%

The PVT engine uses the following property-column names:

  Column             Description                     Unit
  Bo_rb_per_STB    Oil formation volume factor     rb/STB
Bg_rb_per_scf    Gas formation volume factor     rb/scf
Bw_rb_per_STB    Water formation volume factor   rb/STB
Rs_scf_per_STB   Solution gas-oil ratio          scf/STB

A separate laboratory PVT table, when used, should contain
Pressure_psia and the PVT property columns to be interpolated.

Unit consistency is essential. In particular, gas production is
represented as Mscf in the sample reservoir dataset, whereas Bg is
represented in rb/scf. Any downstream reservoir-volume calculation must
convert Mscf to scf (multiply by 1,000) before multiplying by Bg.
Installation
Requirements
- Python 3
- Packages listed in requirements.txt
Setup
Clone the repository:
git clone https://github.com/Rudra03Rath/PyMBAL-An-Intelligent-Material-Balance-Reserve-Estimation-System.git
cd PyMBAL-An-Intelligent-Material-Balance-Reserve-Estimation-System
Create and activate a virtual environment (recommended):
python -m venv .venv
Windows:
.venv\Scripts\activate
macOS / Linux:
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Running the current workflow
The current app.py is a command-line workflow that loads a configured
sample file, validates and cleans the dataset, runs the selected PVT
mode, and prints selected PVT output columns.
1. Open app.py.
2. Set PVT_MODE to one of:
   - "laboratory"
   - "correlation"
   - "hybrid"
3. Set FILEPATH to the input CSV or Excel file.
4. For correlation or hybrid calculations, review the configured
   temperature, API gravity, gas gravity, and salinity values.
5. Run:
python app.py
The current application is not yet a complete material-balance or
reserves-calculation application.
Engineering background
Formation volume factors
Oil formation volume factor:
\[
`\frac{\text{oil volume at reservoir conditions}}{\text{oil volume at stock-tank conditions}}`{=tex}
\]
Gas formation volume factor:
\[
`\frac{\text{gas volume at reservoir conditions}}{\text{gas volume at standard conditions}}`{=tex}
\]
Water formation volume factor:
\[
`\frac{\text{water volume at reservoir conditions}}{\text{water volume at standard conditions}}`{=tex}
\]
Solution gas-oil ratio:
\[
`\frac{\text{solution gas at standard conditions}}{\text{stock-tank oil volume}}`{=tex}
\]
These properties are required to convert surface production volumes to
consistent reservoir-condition volumes and to evaluate fluid expansion.
Material balance concept
The material balance equation is based on conservation of mass. In a
reservoir-scale accounting model, cumulative underground withdrawal is
balanced by the expansion of reservoir fluids, rock/connate-water
effects, and any influx from an aquifer, subject to the selected
reservoir model and its assumptions.

A commonly used general form for a conventional oil reservoir is:

\(F = N(E_o + mE_g\^\* + E\_{fw}) + W_e\)

where the terms represent underground withdrawal, oil/solution-gas
expansion, gas-cap expansion, rock/connate-water expansion, and
cumulative water influx. Definitions and volume bases must be kept
consistent for the chosen formulation.

This equation is included as engineering context. The current public
repository does not yet implement the complete material-balance
workflow.
Planned development roadmap
The following are intended future development stages:
- [ ] Reservoir-state and initial-condition calculations
- [ ] Underground-withdrawal calculations with unit-consistent gas
  accounting
- [ ] Oil-reservoir material balance formulations
- [ ] Havlena--Ode analysis and parameter estimation
- [ ] Gas-cap identification and estimation
- [ ] Aquifer models, including Schilthuis and additional
  transient/finite-aquifer options
- [ ] Drive-mechanism interpretation with supporting evidence
- [ ] OOIP and GIIP estimation workflows
- [ ] Decline Curve Analysis and production forecasting
- [ ] Recovery and remaining-volume evaluation
- [ ] Sensitivity and uncertainty analysis
- [ ] Engineering plots and reporting
- [ ] Expanded validation against benchmark problems and suitable
  field datasets
Current limitations and engineering cautions
- Empirical PVT correlations are approximations and may not represent
  unusual or compositionally complex fluids.
- Linear interpolation is an approximation; PVT behaviour may be
  nonlinear, particularly near phase transitions.
- Extrapolated PVT values can be unreliable and should be flagged.
- A representative average reservoir pressure is essential for
  meaningful material-balance interpretation.
- A single-tank material-balance model may be unsuitable for strongly
  compartmentalized reservoirs or systems with significant pressure
  gradients.
- Aquifer influx, gas-cap size, and OOIP can be difficult to identify
  uniquely from pressure and production history alone.
- Synthetic/sample data are useful for software testing but do not
  establish field accuracy.
- Reserves evaluation requires production forecasts, technical and
  commercial assumptions, and the applicable reserves classification
  framework. An in-place estimate alone is not a reserves estimate.
Future engineering validation should include unit tests, analytical
benchmark cases, sensitivity checks, comparison with trusted
calculations/software, and---where available---representative laboratory
and field data.
Repository structure
PyMBAL-An-Intelligent-Material-Balance-Reserve-Estimation-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_data/
│       ├── sample_material_balance_data.csv
│       ├── sample_correlation_pvt_data.xlsx
│       └── sample_hybrid_pvt_data.xlsx
│
├── src/
│   ├── pvt/
│   │   ├── __init__.py
│   │   ├── correlations.py
│   │   ├── gas.py
│   │   ├── interpolation.py
│   │   ├── oil.py
│   │   ├── pvt_engine.py
│   │   └── water.py
│   │
│   └── utils/
│       ├── cleaner.py
│       ├── loader.py
│       └── validator.py
│
├── test_gas.py
├── test_interpolation.py
├── test_oil.py
├── test_pvt_engine.py
└── test_water.py
Disclaimer
PyMBAL is an academic and placement-oriented petroleum engineering
project. Its outputs should not be used as the sole basis for
field-development, investment, drilling, production, or
reserves-reporting decisions. Engineering calculations require
independent verification, representative input data, suitable model
assumptions, and review by qualified professionals.
