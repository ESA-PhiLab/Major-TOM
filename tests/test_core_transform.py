"""correct_core_transform: Core's stored origin -> the band's pixel grid (offline)."""
from affine import Affine

from majortom.spec import correct_core_transform


def test_rounds_native_origin():
    t = Affine(10, 0, 500_003.7, 0, -10, 6_200_044.9)
    assert correct_core_transform(t, "Core-S2L2A", "EPSG:32633") == Affine(10, 0, 500_000, 0, -10, 6_200_040)


def test_rounds_to_each_band_grid():
    """B01 (60 m) errors reach 30 m; the origin moves to the 60 m grid, not the 10 m one."""
    t = Affine(60, 0, 399_972.0, 0, -60, 5_300_031.0)
    assert correct_core_transform(t, "Core-S2L1C", "EPSG:32633") == Affine(60, 0, 399_960, 0, -60, 5_300_040)


def test_southern_utm():
    """10 m: rounding to multiples of 10 is the same in either hemisphere."""
    t = Affine(10, 0, 612_345.6, 0, -10, 8_505_678.4)
    assert correct_core_transform(t, "Core-S1RTC", "EPSG:32736") == Affine(10, 0, 612_350, 0, -10, 8_505_680)


def test_southern_utm_60m():
    """60 m in a southern zone: Sentinel-2's grid has y = 40 (mod 60) (611D_478L, B01 identical to Core)."""
    t = Affine(60, 0, 300_011.3, 0, -60, 3_928_810.09)
    c = correct_core_transform(t, "Core-S2L2A", "EPSG:32718")
    assert c.f % 60 == 40 and c.c % 60 == 0
    assert c == Affine(60, 0, 300_000, 0, -60, 3_928_840)


def test_keeps_dem():
    t = Affine(30, 0, 500_003.7, 0, -30, 6_200_044.9)
    assert correct_core_transform(t, "Core-DEM", "EPSG:32633") == t


def test_error_within_half_pixel():
    t = Affine(20, 0, 123_459.99, 0, -20, 4_567_890.01)
    c = correct_core_transform(t, "Core-S2L2A", "EPSG:32633")
    assert abs(c.c - t.c) <= 10 and abs(c.f - t.f) <= 10
