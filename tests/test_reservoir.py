from src.core.reservoir import ReservoirModel


CONFIG_PATH = (
    "data/reservoir_config/"
    "reservoir_parameters.json"
)


# ------------------------------------------------------------
# Load reservoir model
# ------------------------------------------------------------

reservoir = ReservoirModel(
    CONFIG_PATH
)


# ------------------------------------------------------------
# Get initial reservoir state
# ------------------------------------------------------------

state = reservoir.get_initial_state()


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

reservoir.display_state(
    state
)