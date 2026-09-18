"""sd-forge-res4lyf: port of the RES4LYF sampler family for forge-neo.

This module is loaded by sd-webui-forge-classic (Neo) at startup.  It does
three things:

1.  Installs a ``comfy`` compatibility shim so the vendored RES4LYF code
    (which targets ComfyUI's API surface) can be imported unchanged.
2.  Registers each RES / DEIS sampler as a ``sample_<name>`` attribute on
    ``k_diffusion.sampling`` and adds matching ``SamplerData`` entries via
    :func:`modules.sd_samplers.add_sampler`.
3.  Registers any RES4LYF-specific schedulers that aren't already exposed
    by forge-neo (the ``bong_tangent`` scheduler ships with forge-neo, so
    we mainly add the ``tan_scheduler`` family).
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Callable

import gradio as gr
from modules import infotext_utils, script_callbacks, scripts, shared
from modules.ui_components import InputAccordion

# Ensure the extension root is on sys.path so `lib_res4lyf` is importable.
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

logger = logging.getLogger("sd-forge-res4lyf")

_CURRENT_SKIP_FINAL: bool | None = None

# ---------------------------------------------------------------------------
# Install comfy compat shim BEFORE importing the vendored code.
# ---------------------------------------------------------------------------
try:
    from lib_res4lyf import _compat  # noqa: F401  (side-effect import)
except Exception as exc:  # pragma: no cover - extension boot guard
    logger.error("sd-forge-res4lyf: failed to install comfy compat shim: %s", exc)
    logger.debug(traceback.format_exc())
    raise


def _safe_import_rk_beta():
    """Import ``rk_sampler_beta`` lazily so any failure surfaces clearly."""
    from lib_res4lyf.beta import rk_sampler_beta  # type: ignore

    return rk_sampler_beta


def _build_wrapper(rk_type: str, ode: bool = False) -> Callable:
    """Return a k-diffusion-style sample function for ``rk_type``."""

    rk_beta = _safe_import_rk_beta()
    sample_rk_beta = rk_beta.sample_rk_beta

    if ode:
        kwargs = dict(rk_type=rk_type, eta=0.0, eta_substep=0.0)
    else:
        kwargs = dict(rk_type=rk_type)

    def _sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **_ignored):
        # Wrap the forge-neo denoiser so the vendored code sees the comfy
        # attribute layout (``model.inner_model.inner_model.model_sampling``
        # etc.).
        adapter = _compat.adapt_denoiser(model)

        # The vendored sampler expects ``extra_args`` to already carry
        # ``model_options['transformer_options']`` for region/mask plumbing.
        extra_args = {} if extra_args is None else dict(extra_args)
        model_options = dict(extra_args.get("model_options") or {})
        transformer_options = dict(model_options.get("transformer_options") or {})
        model_options["transformer_options"] = transformer_options
        extra_args["model_options"] = model_options

        # RES4LYF promotes its internal math to fp64; cast the returned
        # latent back to the dtype forge-neo handed us so the rest of the
        # pipeline (VAE decode, hires fix, etc.) doesn't have to deal with
        # an unexpected fp64 tensor.
        in_dtype = x.dtype
        in_device = x.device

        call_kwargs = dict(kwargs)

        p = getattr(model, "p", None)
        if p is not None and hasattr(p, "_res4lyf_skip_final"):
            skip_final = bool(p._res4lyf_skip_final)
        elif _CURRENT_SKIP_FINAL is not None:
            skip_final = bool(_CURRENT_SKIP_FINAL)
        else:
            try:
                from modules import shared
                skip_final = getattr(shared.opts, "res4lyf_skip_final_model_call", True)
            except Exception:
                skip_final = True

        existing_eo = call_kwargs.get("extra_options", "")
        if skip_final:
            if "skip_final_model_call" not in existing_eo:
                call_kwargs["extra_options"] = f"{existing_eo}\nskip_final_model_call".strip()
        else:
            if "skip_final_model_call" in existing_eo:
                lines = [line for line in existing_eo.splitlines() if line.strip() != "skip_final_model_call"]
                call_kwargs["extra_options"] = "\n".join(lines).strip()

        out = sample_rk_beta(
            adapter,
            x,
            sigmas,
            None,  # sampler arg (unused)
            extra_args,
            callback,
            disable,
            **call_kwargs,
        )
        if out.dtype != in_dtype or out.device != in_device:
            out = out.to(dtype=in_dtype, device=in_device)
        return out

    _sample.__name__ = f"sample_{rk_type}{'_ode' if ode else ''}"
    return _sample


# Definitions mirror RES4LYF/beta/__init__.py with extended RK solvers.
_RES_SAMPLERS = [
    # (label, rk_type, ode, scheduler_hint)
    ("RES 2M",        "res_2m",     False, "beta"),
    ("RES 3M",        "res_3m",     False, "beta"),
    ("RES 2S",        "res_2s",     False, "beta"),
    ("RES 3S",        "res_3s",     False, "beta"),
    ("RES 4S",        "res_4s",     False, "beta"),
    ("RES 5S",        "res_5s",     False, "beta"),
    ("RES 6S",        "res_6s",     False, "beta"),
    ("RES 8S",        "res_8s",     False, "beta"),
    ("RES 2M ODE",    "res_2m",     True,  "beta"),
    ("RES 3M ODE",    "res_3m",     True,  "beta"),
    ("RES 2S ODE",    "res_2s",     True,  "beta"),
    ("RES 3S ODE",    "res_3s",     True,  "beta"),
    ("RES 4S ODE",    "res_4s",     True,  "beta"),
    ("RES 5S ODE",    "res_5s",     True,  "beta"),
    ("RES 6S ODE",    "res_6s",     True,  "beta"),
    ("RES 8S ODE",    "res_8s",     True,  "beta"),
    ("DEIS 2M",       "deis_2m",    False, "beta"),
    ("DEIS 3M",       "deis_3m",    False, "beta"),
    ("DEIS 2M ODE",   "deis_2m",    True,  "beta"),
    ("DEIS 3M ODE",   "deis_3m",    True,  "beta"),
    ("ETDRK 4",       "etdrk4_4s",  False, "beta"),
    ("ETDRK 4 ODE",   "etdrk4_4s",  True,  "beta"),
]


def _register_samplers() -> None:
    try:
        import k_diffusion
        from modules import sd_samplers
        from modules import sd_samplers_common, sd_samplers_kdiffusion
    except Exception as exc:  # pragma: no cover
        logger.error("sd-forge-res4lyf: could not import sampler modules: %s", exc)
        return

    KDiffusionSampler = sd_samplers_kdiffusion.KDiffusionSampler

    registered = 0
    for label, rk_type, ode, scheduler_hint in _RES_SAMPLERS:
        funcname = f"sample_{rk_type}{'_ode' if ode else ''}"
        try:
            sample_func = _build_wrapper(rk_type, ode)
        except Exception as exc:
            logger.error(
                "sd-forge-res4lyf: failed to build wrapper %s: %s", funcname, exc
            )
            logger.debug(traceback.format_exc())
            continue

        setattr(k_diffusion.sampling, funcname, sample_func)

        # The RES samplers manage their own noise samplers internally
        # (see RK_NoiseSampler), so we deliberately do NOT enable
        # ``brownian_noise`` here - otherwise forge-neo would also pass a
        # ``noise_sampler`` kwarg that our wrapper has no use for.
        options = {}
        if scheduler_hint:
            options["scheduler"] = scheduler_hint

        aliases = [funcname, f"k_{funcname}"]
        sampler_label = f"{label} (RES4LYF)"

        # ``sd_samplers_kdiffusion.KDiffusionSampler`` looks up the function
        # via ``getattr(k_diffusion.sampling, funcname)`` so we just pass the
        # string name through.
        data = sd_samplers_common.SamplerData(
            sampler_label,
            lambda model, _fn=funcname: KDiffusionSampler(_fn, model),
            aliases,
            options,
        )
        try:
            sd_samplers.add_sampler(data)
            registered += 1
        except Exception as exc:  # pragma: no cover
            logger.error("sd-forge-res4lyf: add_sampler(%s) failed: %s", sampler_label, exc)
            logger.debug(traceback.format_exc())

    logger.info("sd-forge-res4lyf: registered %d RES4LYF samplers.", registered)


def _register_schedulers() -> None:
    """Add RES4LYF schedulers that forge-neo doesn't already provide."""
    try:
        from modules import sd_schedulers
    except Exception:  # pragma: no cover
        return

    Scheduler = sd_schedulers.Scheduler
    existing = {s.name for s in sd_schedulers.schedulers}

    import math
    import torch

    def _tan_scheduler(n, sigma_min, sigma_max, device, *, pivot=0.6, slope=0.2):
        """RES4LYF tan scheduler (single-stage tangent curve)."""
        steps = n + 2
        slope_eff = slope / (steps / 40.0)
        pivot_step = int(steps * pivot)

        smax = ((2 / math.pi) * math.atan(-slope_eff * (0 - pivot_step)) + 1) / 2
        smin = ((2 / math.pi) * math.atan(-slope_eff * ((steps - 1) - pivot_step)) + 1) / 2
        srange = smax - smin
        sscale = float(sigma_max) - float(sigma_min)

        sigmas = [
            ((((2 / math.pi) * math.atan(-slope_eff * (x - pivot_step)) + 1) / 2) - smin)
            * (1.0 / srange)
            * sscale
            + float(sigma_min)
            for x in range(steps)
        ]
        # Keep n+1 entries to match the forge-neo convention (final entry == 0).
        out = sigmas[:n]
        out.append(0.0)
        return torch.FloatTensor(out).to(device)

    def _beta57_scheduler(n, sigma_min, sigma_max, inner_model, device):
        """RES4LYF ``beta57`` scheduler.

        Upstream implementation::

            comfy.samplers.beta_scheduler(model_sampling, total_steps,
                                          alpha=0.5, beta=0.7)

        i.e. the same Beta-distribution-PPF sigma sampler as forge-neo's
        ``beta``, but with the alpha/beta knobs hard-pinned at the
        RES4LYF defaults (0.5 / 0.7) rather than reading them out of
        ``shared.opts``.
        """
        try:
            import numpy as np
            from scipy import stats
        except Exception:  # pragma: no cover
            logger.error("sd-forge-res4lyf: beta57 needs numpy + scipy")
            raise

        alpha = 0.5
        beta = 0.7

        total_timesteps = len(inner_model.sigmas) - 1
        ts = 1 - np.linspace(0, 1, n, endpoint=False)
        ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

        sigs = []
        last_t = -1
        for t in ts:
            if t != last_t:
                sigs += [float(inner_model.sigmas[int(t)])]
            last_t = t
        sigs += [0.0]
        return torch.FloatTensor(sigs).to(device)

    additions = [
        Scheduler("tan", "Tan (RES4LYF)", _tan_scheduler),
        Scheduler(
            "beta57",
            "Beta57 (RES4LYF)",
            _beta57_scheduler,
            need_inner_model=True,
        ),
    ]

    for sch in additions:
        if sch.name in existing:
            continue
        sd_schedulers.schedulers.append(sch)
        sd_schedulers.schedulers_map[sch.name] = sch
        if sch.label:
            sd_schedulers.schedulers_map[sch.label] = sch


def _clean_extra_options():
    """Ensure res4lyf_skip_final_model_call is not duplicated in extra options section."""
    try:
        from modules import shared
        dirty = False
        for key in ("extra_options_txt2img", "extra_options_img2img"):
            val = getattr(shared.opts, key, None)
            if isinstance(val, list) and "res4lyf_skip_final_model_call" in val:
                setattr(shared.opts, key, [x for x in val if x != "res4lyf_skip_final_model_call"])
                dirty = True
        if dirty:
            try:
                shared.opts.save(shared.config_filename)
            except Exception:
                pass
    except Exception:
        pass


class Res4lyfScript(scripts.Script):
    sorting_priority = 2025

    def title(self):
        return "Skip Final Model Call (RES4LYF)"

    def show(self, is_img2img):
        _clean_extra_options()
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        _clean_extra_options()
        elem_prefix = "img2img" if is_img2img else "txt2img"
        elem_id = f"{elem_prefix}_res4lyf_skip_final"

        with InputAccordion(True, label=self.title(), elem_id=elem_id) as enable:
            gr.Markdown(
                "**Skip final model call at sigma_min (eliminates delay/freeze at 100% / 25/25)**\n\n"
                "Bypasses redundant denoiser evaluation at `sigma_min` for RES, DEIS, and ETDRK samplers, "
                "eliminating the completion delay/freeze before decoded image display."
            )

        def get_skip_final_state(params: dict):
            for k, v in params.items():
                k_clean = str(k).strip().lower()
                if "res4lyf" in k_clean and "skip" in k_clean:
                    return str(v).strip().lower() in ("true", "1", "yes", "on")
                if k_clean in ("res4lyf_skip_final", "res4lyf skip final"):
                    return str(v).strip().lower() in ("true", "1", "yes", "on")
            return gr.skip()

        self.infotext_fields = [
            infotext_utils.PasteField(enable, get_skip_final_state),
        ]

        return [enable]

    def process(self, p, enable: bool = True, *args, **kwargs):
        global _CURRENT_SKIP_FINAL
        _CURRENT_SKIP_FINAL = bool(enable)
        p._res4lyf_skip_final = bool(enable)

    def process_before_every_sampling(self, p, enable: bool = True, *args, **kwargs):
        global _CURRENT_SKIP_FINAL
        _CURRENT_SKIP_FINAL = bool(enable)
        p._res4lyf_skip_final = bool(enable)


def on_ui_settings():
    section = ("res4lyf", "RES4LYF")
    shared.opts.add_option(
        "res4lyf_skip_final_model_call",
        shared.OptionInfo(
            True,
            "Skip final model call at sigma_min (RES4LYF)",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    _clean_extra_options()


try:
    script_callbacks.on_ui_settings(on_ui_settings)
    script_callbacks.on_before_ui(_clean_extra_options)
except Exception:
    pass

_clean_extra_options()


# Run registration at import time.
try:
    _register_samplers()
    _register_schedulers()
except Exception as exc:  # pragma: no cover
    logger.error("sd-forge-res4lyf: registration failed: %s", exc)
    logger.debug(traceback.format_exc())
