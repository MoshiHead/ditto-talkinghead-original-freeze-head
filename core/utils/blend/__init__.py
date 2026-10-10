"""Alpha-blend helper used by `PutBack`.

The fast path is the Cython/C implementation in `blend.pyx` + `blend_impl.c`,
compiled on the fly by `pyximport`. Some environments (notebooks/Colab without
a toolchain, CPUs that reject `-march=native`, Python builds without
`distutils`) cannot build it; in that case we fall back to an equivalent
NumPy implementation so inference still runs, just a little slower.
"""

try:
    import pyximport

    pyximport.install()

    from .blend import blend_images_cy
except Exception as _e:  # pragma: no cover - environment dependent
    import warnings

    import numpy as np

    warnings.warn(
        f"Cython blend extension unavailable ({_e!r}); "
        "falling back to the NumPy implementation."
    )

    def blend_images_cy(mask, frame_warped, frame_rgb, result_buffer):
        """mask: (h, w) float; frame_warped/frame_rgb: (h, w, 3); writes into result_buffer."""
        m = mask[:, :, None]
        blended = m * frame_warped + (1.0 - m) * frame_rgb
        np.clip(blended, 0, 255, out=blended)
        result_buffer[...] = blended.astype(np.uint8)
        return result_buffer


__all__ = ["blend_images_cy"]
