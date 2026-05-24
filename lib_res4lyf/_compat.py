"""ComfyUI compatibility shim for the vendored RES4LYF sampler code.

The upstream code lives inside ComfyUI's custom_nodes ecosystem and freely
imports ``comfy.utils``, ``comfy.model_sampling``, ``comfy.k_diffusion``,
``comfy.model_patcher`` and ``comfy.samplers``.  forge-neo exposes the same
functionality but under different module paths (``backend.utils``,
``backend.modules.k_prediction``, ``modules_forge.packages.k_diffusion``,
``backend.patcher.base``).

This module installs a fake ``comfy`` package into ``sys.modules`` so the
vendored sources can be imported unchanged. It must be imported *before*
any of the ``lib_res4lyf`` submodules.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import torch


# ---------------------------------------------------------------------------
# bislerp - vendored from comfyanonymous/ComfyUI comfy/utils.py
# ---------------------------------------------------------------------------
def _bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    def slerp(b1: torch.Tensor, b2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        c = b1.shape[-1]

        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(1) * b1_normalized + (
            torch.sin(r.squeeze(1) * omega) / so
        ).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old: int, length_new: int, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1)) + 1
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


# ---------------------------------------------------------------------------
# Prediction-type sentinels.
#
# Upstream RES4LYF uses ``isinstance(model_sampling, comfy.model_sampling.EPS)``
# / ``CONST`` to detect epsilon vs. rectified-flow models.  We expose
# duck-typed classes whose ``__instancecheck__`` looks at the
# ``prediction_type`` attribute that forge-neo's ``AbstractPrediction``
# subclasses (and ComfyUI's ``ModelSamplingDiscrete``) both expose.
# ---------------------------------------------------------------------------
def _prediction_type_of(obj: Any) -> str | None:
    pred_type = getattr(obj, "prediction_type", None)
    if isinstance(pred_type, str):
        return pred_type
    cls_name = type(obj).__name__
    if "Flow" in cls_name or "Flux" in cls_name or cls_name == "CONST":
        return "const"
    if "VPred" in cls_name or "VPrediction" in cls_name or cls_name.endswith("V"):
        return "v_prediction"
    if "EDM" in cls_name:
        return "edm"
    return "epsilon"


class _PredictionMeta(type):
    """Meta-class implementing duck-typed ``isinstance`` checks."""

    _matches: tuple[str, ...] = ()

    def __instancecheck__(cls, instance) -> bool:  # noqa: D401
        return _prediction_type_of(instance) in cls._matches


class EPS(metaclass=_PredictionMeta):
    _matches = ("epsilon", "eps")


class V_PREDICTION(metaclass=_PredictionMeta):
    _matches = ("v_prediction", "v")


class EDM(metaclass=_PredictionMeta):
    _matches = ("edm",)


class CONST(metaclass=_PredictionMeta):
    _matches = ("const", "flow", "rectified", "rf")


# Some upstream call sites need a real value, e.g.
# ``comfy.model_sampling.time_snr_shift(alpha, t)``.  Provide a reasonable
# default that simply applies the standard sigmoid-shift formula.
def _time_snr_shift(alpha: float, t):
    import math

    if isinstance(t, torch.Tensor):
        return torch.exp(torch.as_tensor(alpha)) / (
            torch.exp(torch.as_tensor(alpha)) + (1.0 / t - 1.0) ** 1.0
        )
    return math.exp(alpha) / (math.exp(alpha) + (1.0 / t - 1.0) ** 1.0)


# ---------------------------------------------------------------------------
# Adapter that wraps the forge-neo denoiser so that it exposes the
# attributes the vendored sampler expects (``model.inner_model.inner_model
# .model_sampling`` etc.).
# ---------------------------------------------------------------------------
class _DenoiserAdapter:
    """Drop-in proxy that gives the vendored sampler a ComfyUI-shaped model.

    The vendored ``sample_rk_beta`` reaches into the denoiser in several
    places::

        model.inner_model.inner_model.device
        model.inner_model.inner_model.model_sampling
        model.inner_model.inner_model.model_config.unet_config['image_model']
        model.inner_model.predict_noise(...)

    forge-neo exposes the same data but under different paths.  We wrap the
    real denoiser and replace ``.inner_model`` with a proxy that papers
    over the differences.
    """

    def __init__(self, real_model):
        self._real = real_model
        self.__wrapped_inner = _InnerProxy(real_model)

    @property
    def inner_model(self):
        return self.__wrapped_inner

    def __call__(self, *args, **kwargs):
        return self._real(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._real, item)


class _InnerProxy:
    def __init__(self, real_outer):
        # real_outer is the CFGDenoiserKDiffusion. Its .inner_model is a
        # ForgeScheduleLinker exposing `.predictor` and the deeper SD model.
        self._real_outer = real_outer
        self._real_inner = getattr(real_outer, "inner_model", real_outer)

    def __call__(self, *args, **kwargs):
        # When the sampler does ``model.inner_model(*args)`` it expects the
        # ForgeScheduleLinker call signature (forge-neo) or comfy's
        # KSamplerX0Inpaint call signature.  Both behave the same way for
        # the typical (x, sigma, **extra) call used by k-diffusion.
        return self._real_inner(*args, **kwargs)

    # ----- forge-neo attribute remaps ----------------------------------
    @property
    def inner_model(self):
        """Mimic comfy's ``model.inner_model.inner_model`` indirection."""
        # forge-neo path: ``self._real_inner.inner_model`` is shared.sd_model.
        if hasattr(self._real_inner, "inner_model"):
            return _SdModelProxy(self._real_inner)
        return _SdModelProxy(self._real_inner)

    @property
    def predictor(self):
        return getattr(self._real_inner, "predictor", None)

    @property
    def model_sampling(self):
        # Some upstream call sites read ``model.inner_model.model_sampling``
        # directly (without the second hop).  Return the predictor as a
        # stand-in.
        return getattr(self._real_inner, "predictor", None)

    def predict_noise(self, *args, **kwargs):
        return self._real_outer(*args, **kwargs)

    def __getattr__(self, item):
        # Fall through to the real linker first, then to the outer denoiser.
        try:
            return getattr(self._real_inner, item)
        except AttributeError:
            return getattr(self._real_outer, item)


class _SdModelProxy:
    """Proxy exposing ``.device`` / ``.model_sampling`` / ``.model_config``."""

    def __init__(self, linker):
        self._linker = linker
        self._sd_model = getattr(linker, "inner_model", None)

    @property
    def device(self):
        if hasattr(self._sd_model, "device"):
            return self._sd_model.device
        try:
            return next(self._linker.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_sampling(self):
        return getattr(self._linker, "predictor", None)

    @property
    def model_config(self):
        # Stub config object with the keys upstream pokes at.
        return _ModelConfigStub(self._sd_model)

    @property
    def diffusion_model(self):
        """Return the underlying UNet/DiT module, or an empty stub.

        Upstream RES4LYF uses ``model.inner_model.inner_model.diffusion_model``
        for opt-in features (StyleMMDiT, regional guides, eps_out overrides)
        that we don't implement.  Python evaluates the attribute *before*
        ``hasattr(stub, "eps_out")`` and friends, so we have to return
        something — an empty stub keeps the hasattr() checks falsy and the
        opt-in branches skipped.
        """
        if self._sd_model is None:
            return _EmptyStub()
        # Try common forge-neo / WebUI locations for the real diffusion module.
        for path in ("diffusion_model", "model"):
            obj = getattr(self._sd_model, path, None)
            if obj is None:
                continue
            if path == "model":
                obj = getattr(obj, "diffusion_model", obj)
            if hasattr(obj, "forward"):
                return obj
        forge_objects = getattr(self._sd_model, "forge_objects", None)
        if forge_objects is not None:
            unet = getattr(forge_objects, "unet", None)
            if unet is not None:
                inner = getattr(unet, "model", unet)
                inner = getattr(inner, "diffusion_model", inner)
                if hasattr(inner, "forward"):
                    return inner
        return _EmptyStub()

    def __getattr__(self, item):
        if self._sd_model is not None and hasattr(self._sd_model, item):
            return getattr(self._sd_model, item)
        return getattr(self._linker, item)


class _EmptyStub:
    """Empty object that swallows attribute access.

    Used as a placeholder for ``model.inner_model.inner_model.diffusion_model``
    when we don't have a real diffusion module to point at.  Attribute access
    raises ``AttributeError`` so ``hasattr(stub, name)`` returns ``False``.
    """

    __slots__ = ()

    def __bool__(self):
        return False


class _ModelConfigStub:
    def __init__(self, sd_model):
        self._sd_model = sd_model
        # Best-effort image_model name detection so is_video_model() can
        # still answer.
        name = ""
        for attr in ("model_type", "unet_class", "name", "filename_pretty"):
            v = getattr(sd_model, attr, None)
            if isinstance(v, str):
                name = v.lower()
                break
        self.unet_config = {"image_model": name}

    def __getattr__(self, item):
        return None


def adapt_denoiser(model):
    """Return ``model`` wrapped in our compatibility adapter.

    If ``model`` already exposes the ComfyUI attribute layout we return it
    unchanged.  ComfyUI's denoiser exposes ``model.inner_model.inner_model
    .{device, model_sampling, model_config}`` directly; forge-neo's
    ``CFGDenoiserKDiffusion`` does *not* (its deepest layer is
    ``shared.sd_model`` which has ``device`` but no ``model_sampling``).
    Checking ``model_sampling`` is the unambiguous signal.
    """
    try:
        ms = model.inner_model.inner_model.model_sampling  # type: ignore[attr-defined]
        if ms is None:
            return _DenoiserAdapter(model)
        # Also need ``device`` to be reachable for the sampler.
        _ = model.inner_model.inner_model.device  # type: ignore[attr-defined]
        return model
    except Exception:
        return _DenoiserAdapter(model)


# ---------------------------------------------------------------------------
# Install fake comfy.* modules so vendored ``import comfy.X`` statements
# succeed.
# ---------------------------------------------------------------------------
def _install_comfy_stubs() -> None:
    if "comfy" in sys.modules and getattr(sys.modules["comfy"], "_res4lyf_compat", False):
        return

    def _new_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    comfy = _new_module("comfy")
    comfy._res4lyf_compat = True

    # comfy.utils ---------------------------------------------------------
    utils = _new_module("comfy.utils")
    utils.bislerp = _bislerp
    try:
        from backend import utils as forge_utils  # type: ignore

        # Re-export anything forge-neo provides for forward compatibility.
        for attr in dir(forge_utils):
            if attr.startswith("_"):
                continue
            if not hasattr(utils, attr):
                setattr(utils, attr, getattr(forge_utils, attr))
    except Exception:
        pass
    comfy.utils = utils

    # comfy.model_sampling -----------------------------------------------
    msamp = _new_module("comfy.model_sampling")
    msamp.EPS = EPS
    msamp.CONST = CONST
    msamp.V_PREDICTION = V_PREDICTION
    msamp.EDM = EDM
    msamp.time_snr_shift = _time_snr_shift
    comfy.model_sampling = msamp

    # comfy.model_patcher -------------------------------------------------
    patcher = _new_module("comfy.model_patcher")
    try:
        from backend.patcher import base as forge_patcher_base  # type: ignore

        patcher.set_model_options_post_cfg_function = (
            forge_patcher_base.set_model_options_post_cfg_function
        )
    except Exception:
        def _set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
            model_options = dict(model_options)
            model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
            if disable_cfg1_optimization:
                model_options["disable_cfg1_optimization"] = True
            return model_options

        patcher.set_model_options_post_cfg_function = _set_model_options_post_cfg_function
    comfy.model_patcher = patcher

    # comfy.k_diffusion(.sampling) ---------------------------------------
    kd = _new_module("comfy.k_diffusion")
    kd_sampling = _new_module("comfy.k_diffusion.sampling")

    _copied_from = None
    # 1. forge-neo bundles its k-diffusion at modules_forge.packages.k_diffusion
    # 2. Once forge-neo has put that path on sys.path the bare ``k_diffusion``
    #    namespace also resolves to the same package - we try that path too in
    #    case the explicit import failed for some reason.
    # 3. As a last resort, a pip-installed ``k_diffusion`` is used.
    for path in (
        "modules_forge.packages.k_diffusion.sampling",
        "k_diffusion.sampling",
    ):
        try:
            mod = importlib.import_module(path)
        except Exception:
            continue
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            setattr(kd_sampling, attr, getattr(mod, attr))
        _copied_from = path
        break

    if not hasattr(kd_sampling, "BrownianTreeNoiseSampler"):
        # Vendored fallback - the noise sampler is a thin wrapper around
        # torchsde, which forge-neo always installs.  Providing this here
        # means the vendored RES4LYF code keeps working even if the k-
        # diffusion module above couldn't be imported (e.g. during a
        # standalone smoke test outside of forge-neo).
        try:
            import torchsde

            class _BatchedBrownianTree:
                def __init__(self, x, t0, t1, seed=None, **kwargs):
                    self.cpu_tree = kwargs.pop("cpu", True)
                    t0, t1, self.sign = self._sort(t0, t1)
                    w0 = kwargs.pop("w0", None)
                    if w0 is None:
                        w0 = torch.zeros_like(x)
                    self.batched = False
                    if seed is None:
                        seed = (torch.randint(0, 2**63 - 1, ()).item(),)
                    elif isinstance(seed, (tuple, list)):
                        if len(seed) != x.shape[0]:
                            raise ValueError("seed list must match batch size")
                        self.batched = True
                        w0 = w0[0]
                    else:
                        seed = (seed,)
                    if self.cpu_tree:
                        t0, w0, t1 = t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu()
                    self.trees = tuple(
                        torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed
                    )

                @staticmethod
                def _sort(a, b):
                    return (a, b, 1) if a < b else (b, a, -1)

                def __call__(self, t0, t1):
                    t0, t1, sign = self._sort(t0, t1)
                    device, dtype = t0.device, t0.dtype
                    if self.cpu_tree:
                        t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
                    w = torch.stack([tree(t0, t1) for tree in self.trees]).to(
                        device=device, dtype=dtype
                    ) * (self.sign * sign)
                    return w if self.batched else w[0]

            class BrownianTreeNoiseSampler:
                def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
                    self.transform = transform
                    t0 = self.transform(torch.as_tensor(sigma_min))
                    t1 = self.transform(torch.as_tensor(sigma_max))
                    self.tree = _BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

                def __call__(self, sigma, sigma_next):
                    t0 = self.transform(torch.as_tensor(sigma))
                    t1 = self.transform(torch.as_tensor(sigma_next))
                    return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

            kd_sampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSampler
        except Exception:
            pass

    if not hasattr(kd_sampling, "default_noise_sampler"):
        def _default_noise_sampler(x):
            return lambda sigma, sigma_next: torch.randn_like(x)

        kd_sampling.default_noise_sampler = _default_noise_sampler

    if not hasattr(kd_sampling, "to_d"):
        def _to_d(x, sigma, denoised):
            return (x - denoised) / sigma.view(sigma.shape + (1,) * (x.ndim - sigma.ndim))

        kd_sampling.to_d = _to_d

    if not hasattr(kd_sampling, "get_sigmas_karras"):
        def _get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
            ramp = torch.linspace(0, 1, n, device=device)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            return torch.cat([sigmas, sigmas.new_zeros([1])])

        kd_sampling.get_sigmas_karras = _get_sigmas_karras

    if not hasattr(kd_sampling, "get_sigmas_polyexponential"):
        def _get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.0, device="cpu"):
            import math

            ramp = torch.linspace(1, 0, n, device=device) ** rho
            sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
            return torch.cat([sigmas, sigmas.new_zeros([1])])

        kd_sampling.get_sigmas_polyexponential = _get_sigmas_polyexponential

    if not hasattr(kd_sampling, "get_sigmas_exponential"):
        def _get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
            import math

            sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
            return torch.cat([sigmas, sigmas.new_zeros([1])])

        kd_sampling.get_sigmas_exponential = _get_sigmas_exponential

    if not hasattr(kd_sampling, "get_ancestral_step"):
        def _get_ancestral_step(sigma_from, sigma_to, eta=1.0):
            if not eta:
                return sigma_to, 0.0
            sigma_up = min(
                sigma_to,
                eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
            )
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
            return sigma_down, sigma_up

        kd_sampling.get_ancestral_step = _get_ancestral_step

    kd.sampling = kd_sampling
    comfy.k_diffusion = kd

    # comfy.samplers ------------------------------------------------------
    samplers = _new_module("comfy.samplers")
    samplers.SCHEDULER_NAMES = [
        "normal",
        "karras",
        "exponential",
        "sgm_uniform",
        "simple",
        "ddim_uniform",
        "beta",
        "linear_quadratic",
        "kl_optimal",
        "bong_tangent",
    ]

    class _SchedulerHandler:  # placeholder, not used by sample_rk_beta itself
        def __init__(self, handler=None, use_ms=False):
            self.handler = handler
            self.use_ms = use_ms

    samplers.SchedulerHandler = _SchedulerHandler
    samplers.SCHEDULER_HANDLERS = {}
    comfy.samplers = samplers

    # comfy.ldm.common_dit ------------------------------------------------
    ldm = _new_module("comfy.ldm")
    common_dit = _new_module("comfy.ldm.common_dit")
    try:
        from backend.utils import pad_to_patch_size as forge_pad  # type: ignore

        common_dit.pad_to_patch_size = forge_pad
    except Exception:
        def _pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
            import torch.nn.functional as F

            pad_h = (-img.shape[-2]) % patch_size[-2]
            pad_w = (-img.shape[-1]) % patch_size[-1]
            return F.pad(img, (0, pad_w, 0, pad_h), mode=padding_mode)

        common_dit.pad_to_patch_size = _pad_to_patch_size
    ldm.common_dit = common_dit
    comfy.ldm = ldm

    # comfy.supported_models / comfy.sd / etc. - empty stubs --------------
    for sub in (
        "supported_models",
        "sd",
        "sample",
        "sampler_helpers",
        "latent_formats",
        "nested_tensor",
    ):
        if f"comfy.{sub}" not in sys.modules:
            _new_module(f"comfy.{sub}")


_install_comfy_stubs()
