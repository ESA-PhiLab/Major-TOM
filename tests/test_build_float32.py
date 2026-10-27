"""majortom.build.as_float32: float64 bands are stored as float32 and recorded (offline)."""
import numpy as np

from majortom.build import as_float32


def test_float64_becomes_float32_and_is_recorded():
    planes = {"vv": np.array([[-12.123456789012345]], np.float64), "label": np.array([[3]], np.uint8),
              "B04": np.array([[1234]], np.uint16), "t": np.array([[0.5]], np.float32)}
    out, source = as_float32(planes)
    assert {b: str(v.dtype) for b, v in out.items()} == {"vv": "float32", "label": "uint8", "B04": "uint16", "t": "float32"}
    assert source == {"vv": "float64"}
    assert abs(float(out["vv"][0, 0]) - planes["vv"][0, 0]) / 12.12 < 1e-7
