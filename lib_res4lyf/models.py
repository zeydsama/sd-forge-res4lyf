"""Minimal model-sampling type registry used by the vendored RES4LYF code.

The upstream ``models.py`` carries Flux-variant patches and ComfyUI-only
node classes which we cannot port to forge-neo.  The samplers we expose
only need the ``PRED`` taxonomy, so we vendor that tiny helper here.
"""

from typing import Set, Type

from . import _compat

# Re-export prediction-type sentinels from the compat shim.
from ._compat import EPS, CONST, V_PREDICTION, EDM  # noqa: F401


class _Tag(metaclass=_compat._PredictionMeta):
    """Generic prediction-type marker with no matches."""

    _matches: tuple = ()


class X0(_Tag):
    _matches = ("x0",)


class IMG_TO_IMG(_Tag):
    _matches = ("img_to_img", "i2i")


class PRED:
    """Compatibility shim for upstream's :class:`PRED`."""

    TYPE_VP: Set[Type] = {CONST}
    TYPE_VE: Set[Type] = {EPS}
    TYPE_VPRED: Set[Type] = {V_PREDICTION, EDM}
    TYPE_X0: Set[Type] = {X0, IMG_TO_IMG}

    TYPE_ALL = TYPE_VP | TYPE_VE | TYPE_VPRED | TYPE_X0

    @classmethod
    def get_type(cls, model_sampling):
        """Best-effort prediction-type detection.

        Upstream walks ``type(model_sampling).__mro__`` looking for one of
        the sentinel classes.  We expose duck-typed sentinels via the
        ``_PredictionMeta`` meta-class, so a simple ``isinstance`` check
        does the right thing on both ComfyUI and forge-neo predictors.
        """
        for v_type in cls.TYPE_ALL:
            if isinstance(model_sampling, v_type):  # type: ignore[arg-type]
                return v_type
        return None
