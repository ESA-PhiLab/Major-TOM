"""majortom.climate.sources: one real ERA5 field through the cell weights (online, `pytest -m network`)."""
import numpy as np
import pandas as pd
import pytest

from majortom.cells import Cell
from majortom.climate import cell_means, weights
from majortom.climate.sources import SOURCES, field, valid_range

ERA5 = SOURCES["era5"]
LOCH_LOMOND = Cell(624, -29)          # 624U_29L, the builder's test cell
WHEN = pd.Timestamp("2024-07-15T11:00")


@pytest.mark.network
def test_era5_field_at_loch_lomond():
    t2m = field(ERA5, "t2m", WHEN)
    assert t2m.shape == (ERA5.grid.height, ERA5.grid.width) and np.isfinite(t2m).all()
    cells = pd.DataFrame({"grid_row_u": [LOCH_LOMOND.row], "grid_col_r": [LOCH_LOMOND.col]})
    value, valid = cell_means(weights(cells, ERA5.grid), t2m)
    assert valid[0] == pytest.approx(1) and 280 < value[0] < 300            # a summer late morning in K


@pytest.mark.network
def test_era5_refuses_empty_times():
    start, stop = valid_range(ERA5)
    assert start == pd.Timestamp("1940-01-01")
    with pytest.raises(ValueError):
        field(ERA5, "t2m", stop + pd.Timedelta(days=400))


def test_era5_land_monthly_all_names():
    """All 49 ERA5-Land variables, as Earth Data Hub's build, with ERA5-Land's own Climate Data Store names."""
    src = SOURCES["era5-land-monthly-all"]
    assert list(src.variables) == list(SOURCES["era5-land-edh"].variables)
    assert len(set(src.variables.values())) == 49
    assert src.variables["sd"] == "snow_depth_water_equivalent"          # ERA5's "snow_depth" is sd; not here
    assert src.variables["sde"] == "snow_depth"
    assert src.variables["e"] == "total_evaporation"
    assert src.units["tp"] == "m day**-1" and src.units["t2m"] == "K"   # monthly means of accumulations: per day
