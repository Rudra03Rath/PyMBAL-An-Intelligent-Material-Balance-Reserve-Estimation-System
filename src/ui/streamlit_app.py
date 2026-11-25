from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import json
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from app import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    run_pymbal_pipeline,
)


def _plotly_layout(fig, title=None):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        margin=dict(
            l=55,
            r=35,
            t=160,
            b=60,
        ),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),

        title=dict(
            text=title,
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=20),
        )
        if title
        else None,
    )

    fig.update_layout(**layout)

    return fig


def _make_pdf_report(report, input_filename, assumptions):
    """
    Build a compact engineering PDF from the validated PyMBAL report.

    The PDF is a reporting/serialization layer only. It does not perform
    reservoir calculations and explicitly labels technical reserves as
    non-economic.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    summary = report.executive_summary()
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="PyMBAL Engineering Report",
        author="PyMBAL",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PyMBALTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "PyMBALSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        spaceAfter=14,
    )

    section_style = ParagraphStyle(
        "PyMBALSection",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "PyMBALBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    small_style = ParagraphStyle(
        "PyMBALSmall",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
    )

    story = []

    story.append(Paragraph("PyMBAL", title_style))
    story.append(
        Paragraph(
            "Intelligent Material Balance & Reserve Evaluation System",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Engineering Report</b><br/>"
            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}<br/>"
            f"Input dataset: {input_filename}",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # Executive summary
    story.append(Paragraph("1. Executive Summary", section_style))

    summary_rows = [
        ["Parameter", "Result"],
        ["Selected configuration", str(summary["selected_configuration"])],
        [
            "Aquifer model",
            str(summary["selected_aquifer_model"] or "None"),
        ],
        ["Identifiability", str(summary["identifiability_status"])],
        ["Dominant drive", f"{summary['dominant_drive']} ({summary['dominant_drive_percentage']:.2f}%)"],
        ["Overall drive", str(summary["overall_drive"])],
        ["OOIP", f"{summary['ooip_mmstb']:.4f} MMSTB"],
        ["GIIP", f"{summary['giip_bscf']:.4f} Bscf"],
        ["Best DCA", str(summary["selected_dca_model"])],
        ["EUR", f"{summary['eur_mmstb']:.4f} MMSTB"],
        [
            "Remaining technical",
            f"{summary['remaining_reserves_mmstb']:.4f} MMSTB",
        ],
        [
            "Ultimate recovery factor",
            f"{summary['ultimate_recovery_factor_percent']:.2f}%",
        ],
    ]

    table = Table(summary_rows, colWidths=[80 * mm, 120 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8b8b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    # Interpretation
    story.append(Paragraph("2. Reservoir Interpretation", section_style))
    story.append(
        Paragraph(
            f"The selected material-balance interpretation is "
            f"<b>{summary['selected_configuration']}</b>. "
            f"The identified aquifer model is "
            f"<b>{summary['selected_aquifer_model'] or 'None'}</b>. "
            f"The model-selection identifiability status is "
            f"<b>{summary['identifiability_status']}</b>.",
            body_style,
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            f"The dominant calculated drive mechanism is "
            f"<b>{summary['dominant_drive']}</b>, contributing "
            f"<b>{summary['dominant_drive_percentage']:.2f}%</b> of the "
            f"final drive index. The overall drive classification is "
            f"<b>{summary['overall_drive']}</b>.",
            body_style,
        )
    )

    # DDI
    story.append(Paragraph("3. Drive Mechanism", section_style))
    ddi = report.final_ddi()
    ddi_rows = [["Drive mechanism", "Contribution (%)"]]
    ddi_rows.extend(
        [[str(k), f"{float(v):.2f}"] for k, v in ddi.items()]
    )
    ddi_table = Table(ddi_rows, colWidths=[110 * mm, 65 * mm])
    ddi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]
        )
    )
    story.append(ddi_table)

    # Volumetrics
    story.append(Paragraph("4. Volumetric Interpretation", section_style))
    vol = report.volumetric_candidates()
    if not vol.empty:
        display_cols = [
            c for c in [
                "Configuration",
                "Aquifer_Model",
                "OOIP_STB",
                "GIIP_scf",
                "GIIP_Bscf",
                "RMSE_rb",
                "R2",
                "Competitive",
                "Success",
            ]
            if c in vol.columns
        ]
        vol_view = vol[display_cols].copy()

        for col in ["OOIP_STB", "GIIP_scf"]:
            if col in vol_view.columns:
                vol_view[col] = vol_view[col].map(
                    lambda x: f"{float(x):,.0f}" if pd.notna(x) else ""
                )

        vol_rows = [list(vol_view.columns)]
        vol_rows.extend(
            vol_view.fillna("").astype(str).values.tolist()
        )

        vol_table = Table(vol_rows, repeatRows=1)
        vol_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(vol_table)

    # DCA
    story.append(Paragraph("5. Production Forecast & DCA", section_style))
    dca = report.dca_comparison()
    if not dca.empty:
        dca_view = dca.copy()
        if "EUR" in dca_view.columns:
            dca_view["EUR"] = dca_view["EUR"].map(
                lambda x: f"{float(x) / 1e6:.4f}" if pd.notna(x) else ""
            )

        dca_rows = [list(dca_view.columns)]
        dca_rows.extend(
            dca_view.fillna("").astype(str).values.tolist()
        )

        dca_table = Table(dca_rows, repeatRows=1)
        dca_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(dca_table)

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"Mathematical best-fit DCA model: "
            f"<b>{summary['selected_dca_model']}</b>.",
            body_style,
        )
    )

    # Reserves
    story.append(Paragraph("6. Technical Reserves", section_style))
    reserves = report.reserve_candidates()
    if not reserves.empty:
        reserve_cols = [
            c for c in [
                "Model",
                "Technical_EUR_MMSTB",
                "Technical_Reserves_MMSTB",
                "Historical_RF_percent",
                "Ultimate_RF_percent",
                "OOIP_Constrained",
                "Selected",
            ]
            if c in reserves.columns
        ]
        reserve_view = reserves[reserve_cols].copy()

        reserve_rows = [list(reserve_view.columns)]
        reserve_rows.extend(
            reserve_view.fillna("").astype(str).values.tolist()
        )

        reserve_table = Table(reserve_rows, repeatRows=1)
        reserve_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(reserve_table)

    # Warnings / evidence / notes
    story.append(Paragraph("7. Warnings & Evidence", section_style))
    warnings = report.all_warnings()

    if warnings:
        for warning in warnings:
            story.append(Paragraph(f"• {warning}", body_style))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("No engineering warnings were generated.", body_style))

    evidence = report.all_evidence()
    if evidence:
        story.append(Spacer(1, 5))
        story.append(Paragraph("<b>Evidence:</b>", body_style))
        for item in evidence:
            story.append(Paragraph(f"• {item}", small_style))

    if report.notes:
        story.append(Spacer(1, 7))
        story.append(Paragraph("<b>Notes:</b>", body_style))
        for note in report.notes:
            story.append(Paragraph(f"• {note}", small_style))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "<b>Important:</b> Remaining reserves reported by PyMBAL are "
            "technical remaining-recoverable estimates. They are not an "
            "economic-limit analysis and do not constitute SEC, SPE-PRMS, "
            "proved, probable, or possible reserves classification.",
            small_style,
        )
    )

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Forecast assumptions used in this run:</b>", body_style))
    for key, value in assumptions.items():
        story.append(
            Paragraph(
                f"• {key}: {value}",
                small_style,
            )
        )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def main():

    def _render_streamlit():
        """
        Render the first integrated Streamlit UI.

        The UI is deliberately thin: all reservoir calculations happen in
        run_pymbal_pipeline().
        """

        import streamlit as st

        st.set_page_config(
            page_title="PyMBAL",
            page_icon="🛢️",
            layout="wide",
        )

        # Dashboard KPI formatting
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
                font-size: 2rem;
                white-space: nowrap;
            }
            [data-testid="stMetricLabel"] {
                font-size: 1rem;
            }

            .stPlotlyChart {
                border-radius: 15px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title(
            "PyMBAL"
        )

        st.caption(
            "Intelligent Material Balance & Reserve Evaluation System"
        )

        st.sidebar.header("Input")

        uploaded_file = st.sidebar.file_uploader(
            "Reservoir history CSV",
            type=["csv"],
        )

        st.sidebar.header("Reservoir Properties")

        initial_pressure = st.sidebar.number_input(
            "Initial Pressure (psia)",
            min_value=100.0,
            value=4000.0,
            step=100.0,
        )

        initial_temperature = st.sidebar.number_input(
            "Initial Temperature (°F)",
            min_value=50.0,
            value=200.0,
            step=10.0,
        )

        initial_water_saturation = st.sidebar.number_input(
            "Initial Water Saturation",
            min_value=0.01,
            max_value=0.99,
            value=0.25,
            step=0.01,
        )

        formation_compressibility = st.sidebar.number_input(
            "Formation Compressibility (psi⁻¹)",
            min_value=0.0,
            value=6e-6,
            format="%.2e",
        )

        water_compressibility = st.sidebar.number_input(
            "Water Compressibility (psi⁻¹)",
            min_value=0.0,
            value=3e-6,
            format="%.2e",
        )

        st.sidebar.header("Fluid Properties")

        oil_api = st.sidebar.number_input(
            "Oil API Gravity",
            min_value=1.0,
            value=35.0,
            step=1.0,
        )

        gas_gravity = st.sidebar.number_input(
            "Gas Gravity",
            min_value=0.1,
            value=0.8,
            step=0.01,
        )

        water_salinity = st.sidebar.number_input(
            "Water Salinity (wt%)",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

        st.sidebar.header("Gas Cap Input")

        specify_gas_cap_ratio = st.sidebar.checkbox(
            "Specify gas-cap ratio m",
            value=False,
            help="Enable this if the gas-cap ratio m is known from field or reservoir data."
        )

        if specify_gas_cap_ratio:
            gas_cap_ratio_input = st.sidebar.number_input(
                "Gas Cap Ratio m",
                min_value=0.0,
                value=0.80,
                step=0.05,
                format="%.2f",
                help="Known initial gas-cap ratio m."
            )

            gas_cap_ratio = float(gas_cap_ratio_input)

        else:
            gas_cap_ratio = None

            st.sidebar.caption(
                "m will be estimated automatically by PyMBAL."
            )

        st.sidebar.header("Model Constraints")

        gas_cap_choice = st.sidebar.selectbox(
            "Gas-cap presence",
            [
                "Unknown",
                "Present",
                "Absent",
            ],
            index=0,
        )

        aquifer_choice = st.sidebar.selectbox(
            "Aquifer presence",
            [
                "Unknown",
                "Present",
                "Absent",
            ],
            index=0,
        )

        aquifer_model_choice = st.sidebar.selectbox(
            "Aquifer model",
            [
                "Unknown",
                "Schilthuis Steady-State",
                "Fetkovich Finite Aquifer",
                "Van Everdingen-Hurst",
            ],
            index=0,
        )

        st.sidebar.header("Forecast Assumptions")

        forecast_horizon = st.sidebar.number_input(
            "Forecast horizon (days)",
            min_value=30.0,
            value=500.0,
            step=30.0,
        )

        pressure_decline = st.sidebar.slider(
            "Future pressure decline fraction",
            min_value=0.05,
            max_value=0.80,
            value=0.25,
            step=0.05,
        )

        expansion_factor = st.sidebar.number_input(
            "Future expansion endpoint multiplier",
            min_value=1.01,
            max_value=5.0,
            value=1.20,
            step=0.05,
        )

        st.sidebar.divider()

        run_analysis = st.sidebar.button(
            "🚀 Run PyMBAL Analysis",
            use_container_width=True,
        )

        if uploaded_file is None:
            st.info(
                "Upload a reservoir-history CSV from the sidebar to run "
                "the complete PyMBAL pipeline."
            )

            st.markdown(
                """
    ### Pipeline

    **Data → Model Selection → DDI → OOIP/GIIP → Forecast → DCA → Reserves → Report**

    The model-selection layer remains automatic when the corresponding
    reservoir information is left as **Unknown**.
    """
            )
            return

        # --------------------------------------------------------------
        # Uploaded dataset preview
        # --------------------------------------------------------------

        try:
            preview_df = pd.read_csv(uploaded_file)

            st.subheader("Input Dataset Preview")

            c1, c2, c3 = st.columns(3)

            c1.metric("Records", len(preview_df))

            if "Pressure_psia" in preview_df.columns:
                c2.metric(
                    "Initial Pressure",
                    f"{preview_df['Pressure_psia'].iloc[0]:.1f} psi",
                )

                c3.metric(
                    "Final Pressure",
                    f"{preview_df['Pressure_psia'].iloc[-1]:.1f} psi",
                )

            with st.expander("Show uploaded data"):
                st.dataframe(
                    preview_df.head(),
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Unable to preview CSV: {exc}")
            return

        if not run_analysis:
            st.info(
                "Review the inputs and click 'Run PyMBAL Analysis'."
            )
            return

        def _constraint(value):
            if value == "Unknown":
                return None
            return value == "Present"

        gas_cap_present = _constraint(gas_cap_choice)
        aquifer_present = _constraint(aquifer_choice)

        if aquifer_model_choice == "Unknown":
            aquifer_model = None
        else:
            aquifer_model = aquifer_model_choice

        # A specific aquifer model itself implies aquifer presence.
        if aquifer_model is not None:
            aquifer_present = True

        # Temporary files keep existing backend interfaces unchanged.
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
            mode="wb",
        ) as temporary:
            temporary.write(
                uploaded_file.getvalue()
            )
            uploaded_path = temporary.name


        # --------------------------------------------------------------
        # Inject user supplied PVT correlation inputs
        # --------------------------------------------------------------
        # HYBRID PVT mode requires these properties inside the reservoir
        # dataframe for correlation calculations.

        reservoir_df = pd.read_csv(uploaded_path)

        reservoir_df["Temperature_F"] = initial_temperature
        reservoir_df["Oil_API"] = oil_api
        reservoir_df["Gas_Gravity"] = gas_gravity
        reservoir_df["Water_Salinity_wt_pct"] = water_salinity

        reservoir_df.to_csv(
            uploaded_path,
            index=False,
        )

        print("\nDEBUG PVT INPUT COLUMNS")
        print(
            reservoir_df[
                [
                    "Temperature_F",
                    "Oil_API",
                    "Gas_Gravity",
                    "Water_Salinity_wt_pct"
                ]
            ].head()
        )

        config = {
            "reservoir": {
                "initial_pressure_psia": initial_pressure,
                "initial_temperature_F": initial_temperature,
                "initial_water_saturation": initial_water_saturation,
                "formation_compressibility_psi_inv": formation_compressibility,
                "water_compressibility_psi_inv": water_compressibility,
            },
            "fluid": {
                "oil_api": oil_api,
                "gas_gravity": gas_gravity,
                "water_salinity_wt_pct": water_salinity,
            },
        }

        if gas_cap_ratio is not None:
            config["reservoir"]["gas_cap_ratio"] = gas_cap_ratio

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            mode="w",
            delete=False,
            encoding="utf-8",
        ) as config_temp:
            json.dump(config, config_temp, indent=4)
            config_path = config_temp.name

        analysis_key = (
            uploaded_file.name,
            uploaded_file.size,
            initial_pressure,
            initial_temperature,
            initial_water_saturation,
            formation_compressibility,
            water_compressibility,
            oil_api,
            gas_gravity,
            water_salinity,
            gas_cap_ratio,
            gas_cap_choice,
            aquifer_choice,
            aquifer_model_choice,
            forecast_horizon,
            pressure_decline,
            expansion_factor,
        )

        if (
            "pipeline" in st.session_state
            and st.session_state.get("analysis_key") == analysis_key
        ):
            pipeline = st.session_state["pipeline"]
            report = st.session_state.get("report", pipeline.report)

        else:
            try:
                with st.spinner(
                    "Running PyMBAL analytical pipeline..."
                ):
                    pipeline = run_pymbal_pipeline(
                    filepath=uploaded_path,
                    config_path=(
                        Path(config_path)
                        if Path(config_path).is_absolute()
                        else PROJECT_ROOT / config_path
                    ),
                    gas_cap_present=gas_cap_present,
                    aquifer_present=aquifer_present,
                    aquifer_model=aquifer_model,
                    gas_cap_ratio_m=gas_cap_ratio,
                    forecast_horizon_days=forecast_horizon,
                    future_pressure_decline_fraction=pressure_decline,
                        future_expansion_factor=expansion_factor,
                    )

                st.session_state["pipeline"] = pipeline
                st.session_state["report"] = pipeline.report
                st.session_state["analysis_key"] = analysis_key

            except Exception as exc:
                st.error(
                    f"PyMBAL pipeline failed: {exc}"
                )
                st.exception(exc)
                return
            finally:
                try:
                    Path(uploaded_path).unlink(missing_ok=True)
                    Path(config_path).unlink(missing_ok=True)
                except Exception:
                    pass

        if "report" not in locals():
            report = pipeline.report

        summary = report.executive_summary()

        # --------------------------------------------------------------
        # Dashboard Summary
        # --------------------------------------------------------------

        st.subheader("Executive Dashboard")

        kpi = st.columns(5)

        kpi[0].metric(
            "OOIP",
            f"{summary['ooip_stb'] / 1e6:.3f} MMSTB",
        )

        kpi[1].metric(
            "GIIP",
            f"{summary['giip_scf'] / 1e9:.3f} BSCF",
        )

        kpi[2].metric(
            "EUR",
            f"{summary['eur_stb'] / 1e6:.3f} MMSTB",
        )

        kpi[3].metric(
            "Remaining Reserves",
            f"{summary['remaining_reserves_stb'] / 1e6:.3f} MMSTB",
        )

        kpi[4].metric(
            "Best DCA",
            summary["selected_dca_model"],
        )

        st.divider()

        # --------------------------------------------------------------
        # Reservoir Interpretation Cards
        # --------------------------------------------------------------

        st.subheader("Reservoir Interpretation")

        card1, card2, card3 = st.columns(3)

        with card1:
            st.info(
                f"""
**Configuration**

{summary['selected_configuration']}

**Aquifer**

{summary['selected_aquifer_model'] or 'None'}
"""
            )

        with card2:
            st.success(
                f"""
**Dominant Drive**

{summary['dominant_drive']}

Contribution:
{summary['dominant_drive_percentage']:.2f}%

**Overall Drive**

{summary['overall_drive']}
"""
            )

        with card3:
            confidence = summary["identifiability_status"]

            if confidence == "UNIQUE":
                st.success(f"**Confidence**\n\n{confidence}")
            else:
                st.warning(f"**Confidence**\n\n{confidence}")

        # --------------------------------------------------------------
        # Engineering Warnings
        # --------------------------------------------------------------

        st.subheader("Engineering Alerts")

        warnings = report.all_warnings()

        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No engineering warnings detected.")

        if report.is_ambiguous:
            st.warning(
                "Multiple material-balance interpretations remain "
                "competitive. Additional reservoir information may be "
                "required before accepting a unique interpretation."
            )

        # --------------------------------------------------------------
        # Tabs
        # --------------------------------------------------------------

        tabs = st.tabs(
            [
                "Model Selection",
                "Drive Mechanism",
                "Volumetrics",
                "Forecast",
                "DCA",
                "Reserves",
                "Report",
            ]
        )

        with tabs[0]:
            st.subheader("Model Selection Comparison")

            possible = report.possible_candidates()

            st.dataframe(
                possible,
                use_container_width=True,
            )

            if "RMSE" in possible.columns:

                fig = px.bar(
                    possible,
                    x="RMSE",
                    y=possible.index.astype(str),
                    orientation="h",
                    title="Material Balance Model Ranking",
                    text="RMSE",
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Model",
                    xaxis_title="RMSE",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.subheader("Competitive Interpretations")

            st.dataframe(
                report.competitive_candidates(),
                use_container_width=True,
            )

        with tabs[1]:
            st.subheader("Final DDI")

            ddi = report.final_ddi()

            ddi_df = pd.DataFrame(
                {
                    "Drive Mechanism": list(ddi.keys()),
                    "Contribution (%)": list(ddi.values()),
                }
            )

            st.dataframe(
                ddi_df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Final Drive Contribution")

            ddi_chart = pd.DataFrame(
                {
                    "Drive Mechanism": list(ddi.keys()),
                    "Contribution": list(ddi.values()),
                }
            )

            fig = px.pie(
                ddi_chart,
                values="Contribution",
                names="Drive Mechanism",
                hole=0.45,
                title="Drive Mechanism Contribution (%)",
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.subheader("DDI History")
            st.line_chart(
                report.ddi_history()
            )

        with tabs[2]:
            st.subheader("OOIP / GIIP Candidates")
            volumetric = report.volumetric_candidates()

            st.dataframe(
                volumetric,
                use_container_width=True,
            )

            if "OOIP_STB" in volumetric.columns:

                st.subheader("OOIP Comparison")

                ooip_plot = volumetric[
                    ["OOIP_STB"]
                ]

                fig = px.bar(
                    ooip_plot.reset_index(),
                    x=ooip_plot.index.astype(str),
                    y="OOIP_STB",
                    title="OOIP Candidate Comparison",
                    text_auto=".3s",
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Candidate",
                    yaxis_title="OOIP (STB)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


            if "GIIP_scf" in volumetric.columns:

                st.subheader("GIIP Comparison")

                giip_plot = volumetric[
                    ["GIIP_scf"]
                ]

                fig = px.bar(
                    giip_plot.reset_index(),
                    x=giip_plot.index.astype(str),
                    y="GIIP_scf",
                    title="GIIP Candidate Comparison",
                    text_auto=".3s",
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Candidate",
                    yaxis_title="GIIP (scf)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.write(
                f"**Selected OOIP:** "
                f"{report.ooip_stb / 1e6:.4f} MMSTB"
            )

            st.write(
                f"**Selected GIIP:** "
                f"{report.giip_scf / 1e9:.4f} Bscf"
            )

        with tabs[3]:
            st.subheader("Material-Balance Reservoir-State Forecast")

            forecast_history = (
                pipeline.material_balance_forecast.forecast_history
            )

            st.subheader("Pressure Forecast")

            pressure_plot = forecast_history[
                ["Time_days", "Pressure_psia"]
            ]

            fig = px.line(
                pressure_plot,
                x="Time_days",
                y="Pressure_psia",
                title="Reservoir Pressure Forecast",
                markers=True,
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Time (days)",
                yaxis_title="Pressure (psia)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                forecast_history,
                use_container_width=True,
            )

        with tabs[4]:
            st.subheader("DCA Model Comparison")

            dca_comparison = report.dca_comparison()

            st.dataframe(
                dca_comparison,
                use_container_width=True,
                hide_index=True,
            )

            best = report.forecasting.best_dca_result

            dca_metric_cols = st.columns(4)
            dca_metric_cols[0].metric(
                "Selected DCA",
                report.selected_dca_model,
            )
            dca_metric_cols[1].metric(
                "Best RMSE",
                f"{best.rmse:.3f}",
            )
            dca_metric_cols[2].metric(
                "Best R²",
                f"{best.r2:.4f}",
            )
            dca_metric_cols[3].metric(
                "Selected EUR",
                f"{report.eur_stb / 1e6:.3f} MMSTB",
            )

            # ----------------------------------------------------------
            # All three Arps rate forecasts
            # ----------------------------------------------------------

            st.subheader("Three-Arps Production Forecast")

            rate_frames = []

            successful_dca = [
                (model_name, result)
                for model_name, result
                in report.forecasting.dca_results.items()
                if result.success
            ]

            if successful_dca:
                first_model, first_result = successful_dca[0]

                # Historical production is common to all three fits.
                rate_frames.append(
                    pd.DataFrame(
                        {
                            "Time_days": first_result.historical_time,
                            "Rate_STB_day": first_result.historical_rate,
                            "Series": "Historical",
                            "Model": "Historical",
                        }
                    )
                )

            for model_name, result in successful_dca:

                fitted = pd.DataFrame(
                    {
                        "Time_days": result.historical_time,
                        "Rate_STB_day": result.fitted_rate,
                        "Series": "Historical Fit",
                        "Model": model_name,
                    }
                )

                forecast = pd.DataFrame(
                    {
                        "Time_days": result.forecast_time,
                        "Rate_STB_day": result.forecast_rate,
                        "Series": "Forecast",
                        "Model": model_name,
                    }
                )

                rate_frames.extend(
                    [fitted, forecast]
                )

            if rate_frames:
                rate_df = pd.concat(
                    rate_frames,
                    ignore_index=True,
                )

                fig = px.line(
                    rate_df,
                    x="Time_days",
                    y="Rate_STB_day",
                    color="Model",
                    markers=False,
                )

                for trace in fig.data:
                    if trace.name.startswith("Historical"):
                        trace.line.width = 3

                _plotly_layout(
                    fig,
                    title="Exponential vs Harmonic vs Hyperbolic DCA",
                )
                fig.update_layout(
                    xaxis_title="Time (days)",
                    yaxis_title="Oil Rate (STB/day)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # ----------------------------------------------------------
            # Cumulative production / EUR
            # ----------------------------------------------------------

            st.subheader("Cumulative Production & EUR")

            cumulative_frames = []

            for model_name, result in report.forecasting.dca_results.items():

                if not result.success:
                    continue

                cumulative_frames.append(
                    pd.DataFrame(
                        {
                            "Time_days": result.forecast_time,
                            "Cumulative_STB": result.forecast_cumulative,
                            "Model": model_name,
                        }
                    )
                )

            if cumulative_frames:
                cumulative_df = pd.concat(
                    cumulative_frames,
                    ignore_index=True,
                )

                fig = px.line(
                    cumulative_df,
                    x="Time_days",
                    y="Cumulative_STB",
                    color="Model",
                    markers=False,
                )

                _plotly_layout(
                    fig,
                    title="DCA Cumulative Production / EUR",
                )
                fig.update_layout(
                    xaxis_title="Time (days)",
                    yaxis_title="Cumulative Oil (STB)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # ----------------------------------------------------------
            # DCA parameter comparison
            # ----------------------------------------------------------

            st.subheader("DCA Parameter Comparison")

            parameter_cols = [
                c for c in [
                    "Model",
                    "qi",
                    "Di",
                    "b",
                    "RMSE",
                    "MAE",
                    "R2",
                    "EUR",
                ]
                if c in dca_comparison.columns
            ]

            if parameter_cols:

                display_dca = dca_comparison[parameter_cols].copy()

                if "qi" in display_dca.columns:
                    display_dca["qi"] = display_dca["qi"].map(
                        lambda x: f"{x:,.2f}"
                    )

                if "Di" in display_dca.columns:
                    display_dca["Di"] = display_dca["Di"].map(
                        lambda x: f"{x:.6f}"
                    )

                if "EUR" in display_dca.columns:
                    display_dca["EUR"] = display_dca["EUR"].map(
                        lambda x: f"{x/1e6:.3f} MMSTB"
                    )

                st.dataframe(
                    display_dca,
                    use_container_width=True,
                    hide_index=True,
                )

        with tabs[5]:
            st.subheader("Technical Reserves")

            reserve_candidates = report.reserve_candidates()

            st.dataframe(
                reserve_candidates,
                use_container_width=True,
                hide_index=True,
            )

            reserve_metrics = st.columns(4)

            reserve_metrics[0].metric(
                "Historical Production",
                f"{report.reserves.historical_np_stb / 1e6:.3f} MMSTB",
            )
            reserve_metrics[1].metric(
                "Selected EUR",
                f"{report.eur_stb / 1e6:.3f} MMSTB",
            )
            reserve_metrics[2].metric(
                "Remaining Technical",
                f"{report.remaining_reserves_stb / 1e6:.3f} MMSTB",
            )
            reserve_metrics[3].metric(
                "Ultimate RF",
                f"{report.reserves.ultimate_recovery_factor_percent:.2f}%",
            )

            if not reserve_candidates.empty:

                # EUR comparison
                if "Technical_EUR_MMSTB" in reserve_candidates.columns:
                    st.subheader("EUR Comparison")

                    eur_df = reserve_candidates[
                        ["Model", "Technical_EUR_MMSTB"]
                    ].copy()

                    eur_df = eur_df.dropna(
                        subset=["Technical_EUR_MMSTB"]
                    )

                    if not eur_df.empty:
                        fig = px.bar(
                            eur_df,
                            x="Technical_EUR_MMSTB",
                            y="Model",
                            orientation="h",
                            text="Technical_EUR_MMSTB",
                            title="Technical EUR by DCA Model",
                        )

                        fig.update_traces(
                            texttemplate="%{text:.3f} MMSTB",
                            textposition="outside",
                        )

                        _plotly_layout(fig)
                        fig.update_layout(
                            xaxis_title="EUR (MMSTB)",
                            yaxis_title="DCA Model",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                # Remaining reserves comparison
                if "Technical_Reserves_MMSTB" in reserve_candidates.columns:
                    st.subheader("Remaining Technical Reserves Comparison")

                    reserves_df = reserve_candidates[
                        ["Model", "Technical_Reserves_MMSTB"]
                    ].copy()

                    reserves_df = reserves_df.dropna(
                        subset=["Technical_Reserves_MMSTB"]
                    )

                    if not reserves_df.empty:
                        fig = px.bar(
                            reserves_df,
                            x="Technical_Reserves_MMSTB",
                            y="Model",
                            orientation="h",
                            text="Technical_Reserves_MMSTB",
                            title="Remaining Technical Reserves by DCA Model",
                        )

                        fig.update_traces(
                            texttemplate="%{text:.3f} MMSTB",
                            textposition="outside",
                        )

                        _plotly_layout(fig)
                        fig.update_layout(
                            xaxis_title="Remaining Technical (MMSTB)",
                            yaxis_title="DCA Model",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

            st.info(
                "These are technical remaining-recoverable estimates. "
                "They are not economic reserves or reserves-classification results."
            )

        with tabs[6]:
            st.subheader("Engineering Report")

            report_summary = report.executive_summary()

            report_columns = st.columns(4)

            report_columns[0].metric(
                "OOIP",
                f"{report_summary['ooip_mmstb']:.3f} MMSTB",
            )
            report_columns[1].metric(
                "GIIP",
                f"{report_summary['giip_bscf']:.3f} Bscf",
            )
            report_columns[2].metric(
                "EUR",
                f"{report_summary['eur_mmstb']:.3f} MMSTB",
            )
            report_columns[3].metric(
                "Remaining Technical",
                f"{report_summary['remaining_reserves_mmstb']:.3f} MMSTB",
            )

            st.markdown("---")

            st.markdown("### Reservoir Interpretation")

            interpretation = pd.DataFrame(
                {
                    "Parameter": [
                        "Selected configuration",
                        "Aquifer model",
                        "Identifiability",
                        "Dominant drive",
                        "Overall drive",
                        "Best DCA",
                    ],
                    "Result": [
                        report_summary["selected_configuration"],
                        report_summary["selected_aquifer_model"] or "None",
                        report_summary["identifiability_status"],
                        (
                            f"{report_summary['dominant_drive']} "
                            f"({report_summary['dominant_drive_percentage']:.2f}%)"
                        ),
                        report_summary["overall_drive"],
                        report_summary["selected_dca_model"],
                    ],
                }
            )

            st.dataframe(
                interpretation,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Recovery & Reserves")

            recovery = pd.DataFrame(
                {
                    "Metric": [
                        "Historical production",
                        "Historical recovery factor",
                        "Ultimate recovery factor",
                        "EUR",
                        "Remaining technical reserves",
                    ],
                    "Value": [
                        f"{report.reserves.historical_np_stb / 1e6:.4f} MMSTB",
                        f"{report.reserves.historical_recovery_factor_percent:.2f}%",
                        f"{report.reserves.ultimate_recovery_factor_percent:.2f}%",
                        f"{report.eur_stb / 1e6:.4f} MMSTB",
                        f"{report.remaining_reserves_stb / 1e6:.4f} MMSTB",
                    ],
                }
            )

            st.dataframe(
                recovery,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Warnings")

            warnings = report.all_warnings()

            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("No engineering warnings detected.")

            if report.is_ambiguous:
                st.warning(
                    "The model-selection result remains ambiguous. "
                    "The mathematical best case should not be treated as "
                    "uniquely proven without independent reservoir evidence."
                )

            st.markdown("### Evidence & Notes")

            evidence = report.all_evidence()
            notes = report.notes

            if evidence:
                for item in evidence:
                    st.info(item)

            if notes:
                for note in notes:
                    st.info(note)

            st.markdown("---")

            st.markdown("### Export")

            assumptions = {
                "Initial pressure": f"{initial_pressure:.1f} psia",
                "Initial temperature": f"{initial_temperature:.1f} °F",
                "Initial water saturation": f"{initial_water_saturation:.3f}",
                "Formation compressibility": f"{formation_compressibility:.3e} psi⁻¹",
                "Water compressibility": f"{water_compressibility:.3e} psi⁻¹",
                "Oil API": f"{oil_api:.2f}",
                "Gas gravity": f"{gas_gravity:.3f}",
                "Water salinity": f"{water_salinity:.2f} wt%",
                "Gas-cap ratio m": (
                    "Estimated by model"
                    if gas_cap_ratio is None
                    else f"{gas_cap_ratio:.4f}"
                ),
                "Gas-cap constraint": gas_cap_choice,
                "Aquifer constraint": aquifer_choice,
                "Aquifer model constraint": aquifer_model_choice,
                "Forecast horizon": f"{forecast_horizon:.0f} days",
                "Future pressure decline": f"{pressure_decline:.2%}",
                "Future expansion multiplier": f"{expansion_factor:.3f}",
            }

            col_json, col_pdf = st.columns(2)

            json_payload = json.dumps(
                report.to_dict(),
                indent=2,
                default=str,
            )

            col_json.download_button(
                label="⬇️ Download JSON Report",
                data=json_payload,
                file_name="pymbal_engineering_report.json",
                mime="application/json",
                use_container_width=True,
                on_click="ignore",
            )

            try:
                pdf_bytes = _make_pdf_report(
                    report=report,
                    input_filename=uploaded_file.name,
                    assumptions=assumptions,
                )

                col_pdf.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="pymbal_engineering_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    on_click="ignore",
                )
            except Exception as exc:
                col_pdf.error(
                    f"PDF generation unavailable: {exc}"
                )

            with st.expander("Structured report data"):
                st.json(report.to_dict())

    _render_streamlit()


if __name__ == "__main__":
    main()
