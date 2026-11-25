"""
PyMBAL - Gas-Cap Ratio Calibration

Calibrates gas-cap ratio m and OOIP from material-balance history.

Material balance:

    F - We = N(Eo + mEg + Efw)

Rearranged:

    F - We = N(Eo + Efw) + (N*m)Eg

Define:

    A = N
    B = N*m

Then:

    F - We = A(Eo + Efw) + B*Eg

This is solved using linear least squares.

Finally:

    N = A
    m = B / A

This formulation is preferable to directly optimizing m because it
avoids unnecessary nonlinear optimization and makes the N-m coupling
explicit.
"""

import numpy as np
from dataclasses import dataclass


# ======================================================================
# RESULT OBJECT
# ======================================================================

@dataclass
class GasCapCalibrationResult:

    success: bool
    model_name: str

    gas_cap_ratio_m: float
    estimated_ooip_stb: float

    rmse_rb: float
    mae_rb: float
    r2: float

    residuals_rb: np.ndarray
    apparent_ooip_stb: np.ndarray

    message: str


# ======================================================================
# CALIBRATOR
# ======================================================================

class GasCapCalibrator:

    MODEL_NAME = "Gas-Cap Ratio Calibration"

    def __init__(
        self,
        F_rb,
        Eo_rb_per_STB,
        Eg_rb_per_STB,
        Efw_rb_per_STB,
        We_rb,
    ):

        self.F_rb = np.asarray(F_rb, dtype=float)
        self.Eo_rb_per_STB = np.asarray(
            Eo_rb_per_STB,
            dtype=float,
        )
        self.Eg_rb_per_STB = np.asarray(
            Eg_rb_per_STB,
            dtype=float,
        )
        self.Efw_rb_per_STB = np.asarray(
            Efw_rb_per_STB,
            dtype=float,
        )
        self.We_rb = np.asarray(
            We_rb,
            dtype=float,
        )

        self._validate_inputs()

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def _validate_inputs(self):

        arrays = [
            self.F_rb,
            self.Eo_rb_per_STB,
            self.Eg_rb_per_STB,
            self.Efw_rb_per_STB,
            self.We_rb,
        ]

        lengths = [len(x) for x in arrays]

        if len(set(lengths)) != 1:
            raise ValueError(
                "All calibration input arrays must have identical lengths."
            )

        if len(self.F_rb) < 3:
            raise ValueError(
                "At least 3 material-balance data points are required."
            )

        for array in arrays:

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "Calibration inputs must contain only finite values."
                )

        if np.any(self.F_rb < 0):

            raise ValueError(
                "Underground withdrawal F cannot contain negative values."
            )

        if np.any(self.We_rb < 0):

            raise ValueError(
                "Water influx We cannot contain negative values."
            )

        if np.any(self.Eo_rb_per_STB < 0):

            raise ValueError(
                "Oil expansion Eo cannot contain negative values."
            )

        if np.any(self.Eg_rb_per_STB < 0):

            raise ValueError(
                "Gas-cap expansion Eg cannot contain negative values."
            )

        if np.any(self.Efw_rb_per_STB < 0):

            raise ValueError(
                "Formation/water expansion Efw cannot contain negative values."
            )

        if not np.all(
            np.diff(self.F_rb) >= -1e-12
        ):

            raise ValueError(
                "Underground withdrawal F must be non-decreasing."
            )

        if not np.all(
            np.diff(self.We_rb) >= -1e-12
        ):

            raise ValueError(
                "Water influx We must be non-decreasing."
            )

        if abs(self.F_rb[0]) > 1e-10:

            raise ValueError(
                "The first underground withdrawal value must be zero."
            )

        if abs(self.We_rb[0]) > 1e-10:

            raise ValueError(
                "The first water-influx value must be zero."
            )

    # ==================================================================
    # LINEAR SYSTEM
    # ==================================================================

    def _build_linear_system(self):

        """
        Build:

            y = A*x

        where:

            y = F - We

            x = [N, N*m]

            A[:,0] = Eo + Efw_base
            A[:,1] = Eg + Efw_base
        """

        y = self.F_rb - self.We_rb

        expansion_without_gascap = (
            self.Eo_rb_per_STB
            + self.Efw_rb_per_STB
        )

        gas_cap_increment = (
            self.Eg_rb_per_STB
            + self.Efw_rb_per_STB
        )

        matrix = np.column_stack(
            [
                expansion_without_gascap,
                gas_cap_increment,
            ]
        )

        return matrix, y

    # ==================================================================
    # IDENTIFIABILITY CHECK
    # ==================================================================

    def _check_identifiability(self, matrix):

        """
        Check whether the two expansion columns contain independent
        information.

        If:

            Eo + Efw

        and:

            Eg

        are linearly dependent, N and m cannot be uniquely recovered.
        """

        rank = np.linalg.matrix_rank(matrix)

        if rank < 2:

            raise ValueError(
                "Gas-cap ratio and OOIP are not uniquely identifiable "
                "because the supplied expansion histories are linearly "
                "dependent."
            )

    # ==================================================================
    # CALIBRATION
    # ==================================================================

    def calibrate(
        self,
        m_bounds=(0.0, 10.0),
    ):

        if len(m_bounds) != 2:

            raise ValueError(
                "m_bounds must contain exactly two values."
            )

        m_min = float(m_bounds[0])
        m_max = float(m_bounds[1])

        if m_min < 0:

            raise ValueError(
                "Lower gas-cap ratio bound cannot be negative."
            )

        if m_max <= m_min:

            raise ValueError(
                "Upper gas-cap ratio bound must exceed lower bound."
            )

        matrix, y = self._build_linear_system()

        self._check_identifiability(matrix)

        # --------------------------------------------------------------
        # Solve linear least-squares system
        # --------------------------------------------------------------

        coefficients, _, rank, _ = np.linalg.lstsq(
            matrix,
            y,
            rcond=None,
        )

        if rank < 2:

            raise ValueError(
                "Gas-cap calibration system is rank deficient."
            )

        N = float(coefficients[0])

        Nm = float(coefficients[1])

        # --------------------------------------------------------------
        # Physical validation
        # --------------------------------------------------------------

        if not np.isfinite(N) or not np.isfinite(Nm):

            return self._failed_result(
                "Calibration produced non-finite parameters."
            )

        if N <= 0:

            return self._failed_result(
                "Estimated OOIP is non-positive."
            )

        m = Nm / N

        if not np.isfinite(m):

            return self._failed_result(
                "Estimated gas-cap ratio is non-finite."
            )

        # --------------------------------------------------------------
        # Numerical tolerance around physical bounds
        # --------------------------------------------------------------
        #
        # In a true m = 0 case, floating-point least-squares arithmetic
        # may return a tiny negative value such as -1e-15 instead of
        # exactly zero. This is numerical noise, not a physical result.
        #

        BOUND_TOL = 1e-10 * max(
            1.0,
            abs(m_min),
            abs(m_max),
        )

        # Treat numerically negligible gas-cap ratios as exactly zero.
        if abs(m) < BOUND_TOL:
            m = 0.0

        if m < m_min:

            if m >= m_min - BOUND_TOL:

                m = m_min

            else:

                return self._failed_result(
                    "Estimated gas-cap ratio lies below the supplied "
                    "calibration bounds."
                )

        if m > m_max:

            if m <= m_max + BOUND_TOL:

                m = m_max

            else:

                return self._failed_result(
                    "Estimated gas-cap ratio lies above the supplied "
                    "calibration bounds."
                )

        # --------------------------------------------------------------
        # Reconstruct material balance
        # --------------------------------------------------------------

        predicted_F_minus_We = (
            N
            * (
                self.Eo_rb_per_STB
                + m * self.Eg_rb_per_STB
                + (1.0 + m) * self.Efw_rb_per_STB
            )
        )

        actual_F_minus_We = self.F_rb - self.We_rb

        residuals = (
            actual_F_minus_We
            - predicted_F_minus_We
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(residuals)
            )
        )

        ss_res = float(
            np.sum(
                residuals ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    actual_F_minus_We
                    - np.mean(actual_F_minus_We)
                ) ** 2
            )
        )

        if ss_tot > 0:

            r2 = 1.0 - ss_res / ss_tot

        else:

            r2 = 1.0

        # --------------------------------------------------------------
        # Apparent OOIP history
        # --------------------------------------------------------------

        denominator = (
            self.Eo_rb_per_STB
            + m * self.Eg_rb_per_STB
            + (1.0 + m) * self.Efw_rb_per_STB
        )

        apparent_N = np.full(
            len(denominator),
            np.nan,
            dtype=float,
        )

        valid = denominator > 1e-14

        apparent_N[valid] = (
            actual_F_minus_We[valid]
            / denominator[valid]
        )

        return GasCapCalibrationResult(
            success=True,
            model_name=self.MODEL_NAME,

            gas_cap_ratio_m=m,
            estimated_ooip_stb=N,

            rmse_rb=rmse,
            mae_rb=mae,
            r2=float(r2),

            residuals_rb=residuals,
            apparent_ooip_stb=apparent_N,

            message=(
                "Gas-cap ratio and OOIP calibration "
                "completed successfully."
            ),
        )

    # ==================================================================
    # FAILED RESULT
    # ==================================================================

    def _failed_result(self, message):

        return GasCapCalibrationResult(
            success=False,
            model_name=self.MODEL_NAME,

            gas_cap_ratio_m=np.nan,
            estimated_ooip_stb=np.nan,

            rmse_rb=np.inf,
            mae_rb=np.inf,
            r2=-np.inf,

            residuals_rb=np.array([]),
            apparent_ooip_stb=np.array([]),

            message=message,
        )


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def calibrate_gas_cap_ratio(
    F_rb,
    Eo_rb_per_STB,
    Eg_rb_per_STB,
    Efw_rb_per_STB,
    We_rb,
    m_bounds=(0.0, 10.0),
):

    calibrator = GasCapCalibrator(
        F_rb=F_rb,
        Eo_rb_per_STB=Eo_rb_per_STB,
        Eg_rb_per_STB=Eg_rb_per_STB,
        Efw_rb_per_STB=Efw_rb_per_STB,
        We_rb=We_rb,
    )

    return calibrator.calibrate(
        m_bounds=m_bounds
    )


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PyMBAL - GAS CAP RATIO CALIBRATION ENGINE")
    print("=" * 70)

    print()
    print("Module loaded successfully.")

    print()
    print("Formulation:")
    print(
        "  F - We = N(Eo + mEg + Efw)"
    )

    print()
    print("Linearized formulation:")
    print(
        "  F - We = N(Eo + Efw_base) + (Nm)(Eg + Efw_base)"
    )