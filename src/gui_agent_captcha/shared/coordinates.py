"""Coordinate conversion shared by browser and rendered CAPTCHA domains."""

RELATIVE_COORDINATE_MIN = 0.0
RELATIVE_COORDINATE_MAX = 1000.0


def relative_bin_to_pixel_xy(
    x: float,
    y: float,
    size_px: tuple[int, int],
) -> tuple[float, float]:
    """Map Qwen3-VL 0-1000 relative coordinate bins to viewport pixels."""
    width, height = size_px
    if width <= 0 or height <= 0:
        return float(x), float(y)
    x_bin = min(max(float(x), RELATIVE_COORDINATE_MIN), RELATIVE_COORDINATE_MAX)
    y_bin = min(max(float(y), RELATIVE_COORDINATE_MIN), RELATIVE_COORDINATE_MAX)
    return (
        min((x_bin / RELATIVE_COORDINATE_MAX) * width, float(width - 1)),
        min((y_bin / RELATIVE_COORDINATE_MAX) * height, float(height - 1)),
    )
