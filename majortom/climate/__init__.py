"""Weather and climate per grid cell (docs/CLIMATE.md). So far: cell weights, ERA5 fields (`sources`) and the
hourly and daily tables around acquisitions (`series`)."""
from .series import series
from .weights import Grid, Weights, cell_means, weights

__all__ = ["Grid", "Weights", "cell_means", "series", "weights"]
