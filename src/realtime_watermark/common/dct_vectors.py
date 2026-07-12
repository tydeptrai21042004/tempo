from __future__ import annotations

# Standard zig-zag ranks 8 through 39 of an 8x8 DCT.  This 32-dimensional
# vector avoids the DC and lowest frequencies while not relying only on the
# highest coefficients that JPEG removes first.
MID_FREQUENCY_32 = (
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
)

# Standard zig-zag ranks 36 through 63.  This 28-dimensional upper-mid/high
# frequency vector gives linear ISS a longer spreading sequence than the old
# 19-dimensional adapter while avoiding the lowest, host-dominant DCT terms.
UPPER_HIGH_FREQUENCY_28 = (
    (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
    (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3),
    (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6), (6, 5),
    (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
)

__all__ = ["MID_FREQUENCY_32", "UPPER_HIGH_FREQUENCY_28"]
