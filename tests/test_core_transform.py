"""correct_core_transform: Core's stored origin -> the band's pixel grid (offline)."""
from affine import Affine

from majortom.spec import correct_core_transform


def test_rounds_native_origin():
    t = Affine(10, 0, 500_003.7, 0, -10, 6_200_044.9)
    assert correct_core_transform(t, "Core-S2L2A") == Affine(10, 0, 500_000, 0, -10, 6_200_040)


def test_rounds_to_each_band_grid():
    """B01 (60 m) errors reach 30 m; the origin moves to the 60 m grid, not the 10 m one."""
    t = Affine(60, 0, 399_972.0, 0, -60, 5_300_031.0)
    assert correct_core_transform(t, "Core-S2L1C") == Affine(60, 0, 399_960, 0, -60, 5_300_040)


def test_southern_utm():
    """Northings with the 10,000 km false northing are multiples of every pixel size."""
    t = Affine(10, 0, 612_345.6, 0, -10, 8_505_678.4)
    assert correct_core_transform(t, "Core-S1RTC") == Affine(10, 0, 612_350, 0, -10, 8_505_680)


def test_keeps_dem():
    t = Affine(30, 0, 500_003.7, 0, -30, 6_200_044.9)
    assert correct_core_transform(t, "Core-DEM") == t


def test_error_within_half_pixel():
    t = Affine(20, 0, 123_459.99, 0, -20, 4_567_890.01)
    c = correct_core_transform(t, "Core-S2L2A")
    assert abs(c.c - t.c) <= 10 and abs(c.f - t.f) <= 10
