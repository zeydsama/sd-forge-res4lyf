# sd-forge-res4lyf

A port of the [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) sampler
family for [sd-webui-forge-classic (Neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

The upstream project is a ComfyUI extension. This package vendors just the
parts that make sense outside of a ComfyUI node-graph (the underlying RES /
DEIS samplers plus a few of the RES4LYF schedulers) and exposes them as
first-class entries in forge-neo's sampler and scheduler dropdowns.

## Installation

1. Place the entire `sd-forge-res4lyf` folder into your forge-neo
   `extensions/` directory:

       <your-forge-neo-install>/extensions/sd-forge-res4lyf/
       <your-forge-neo-install>/extensions/sd-forge-res4lyf/scripts/forge_res4lyf.py
       <your-forge-neo-install>/extensions/sd-forge-res4lyf/lib_res4lyf/...

2. Restart the WebUI. The new samplers and schedulers appear in the
   standard dropdowns - no extra UI accordion is added.

## What's included

### Samplers

The following RES4LYF samplers are registered under the
**Sampling method** dropdown (suffix `(RES4LYF)`):

| RES4LYF name | Forge label         |
|--------------|---------------------|
| `res_2m`     | `RES 2M (RES4LYF)`      |
| `res_3m`     | `RES 3M (RES4LYF)`      |
| `res_2s`     | `RES 2S (RES4LYF)`      |
| `res_3s`     | `RES 3S (RES4LYF)`      |
| `res_5s`     | `RES 5S (RES4LYF)`      |
| `res_6s`     | `RES 6S (RES4LYF)`      |
| `res_2m_ode` | `RES 2M ODE (RES4LYF)`  |
| `res_3m_ode` | `RES 3M ODE (RES4LYF)`  |
| `res_2s_ode` | `RES 2S ODE (RES4LYF)`  |
| `res_3s_ode` | `RES 3S ODE (RES4LYF)`  |
| `res_5s_ode` | `RES 5S ODE (RES4LYF)`  |
| `res_6s_ode` | `RES 6S ODE (RES4LYF)`  |
| `deis_2m`    | `DEIS 2M (RES4LYF)`     |
| `deis_3m`    | `DEIS 3M (RES4LYF)`     |
| `deis_2m_ode`| `DEIS 2M ODE (RES4LYF)` |
| `deis_3m_ode`| `DEIS 3M ODE (RES4LYF)` |

All samplers use forge-neo's `CFGDenoiser`, so CFG, ADM/IP-Adapter,
ControlNet, LoRA stack, etc. all work just like with the built-in
samplers.

### Schedulers

- **`bong_tangent`** - forge-neo already ships this scheduler upstream
  (originally taken from RES4LYF), so this extension does not add it
  again. Pick `Bong Tangent` from the schedule dropdown.
- **`tan` / `Tan (RES4LYF)`** - the single-stage tangent scheduler from
  RES4LYF.
- **`beta57` / `Beta57 (RES4LYF)`** - the Beta-distribution-PPF sigma
  schedule with `alpha=0.5, beta=0.7` hard-pinned at the RES4LYF
  defaults.  Same math family as forge-neo's built-in `Beta` scheduler,
  but doesn't read `shared.opts.beta_dist_alpha/beta_dist_beta` so the
  curve is stable across UI option changes.

## What's NOT included (and why)

Most of the RES4LYF project is built around **ComfyUI custom-node
classes** that have no equivalent in forge-neo's Gradio-based UI:

- ClownSampler / SharkSampler nodes
- ClownsharKSampler chain, ClownGuides, ClownGuide groups
- Regional / temporal conditioning nodes (`AttnMask`, `RegContext`)
- Style transfer / WCT / Retrojector / `StyleMMDiT` nodes
- Flux variant patches (`ReFlux`, `ReWan`, etc.)
- Latent manipulation nodes (LagrangeInterp, Sortpicker, ...)
- All `Adv` UI dialogs and ComfyUI-side `js/` widgets

Porting these would mean rebuilding forge-neo's UI around RES4LYF's
node graph - well beyond the scope of "sampler + scheduler" parity.
The **mathematical core** of the project (the Runge-Kutta sampler and
its RES / DEIS coefficient tables) is what's exposed here.

## How the port works

The vendored RES4LYF source under `lib_res4lyf/` is **unchanged from
upstream**. A compatibility shim (`lib_res4lyf/_compat.py`) installs a
fake `comfy` package into `sys.modules` so that the upstream
`import comfy.X` statements resolve against forge-neo equivalents:

| Upstream import                          | Resolved to                                            |
|------------------------------------------|--------------------------------------------------------|
| `comfy.utils.bislerp`                    | vendored from ComfyUI                                  |
| `comfy.model_sampling.EPS / CONST / ...` | duck-typed classes that match on `prediction_type`     |
| `comfy.model_patcher.set_model_options_post_cfg_function` | `backend.patcher.base.set_model_options_post_cfg_function` |
| `comfy.k_diffusion.sampling`             | `modules_forge.packages.k_diffusion.sampling`          |
| `comfy.ldm.common_dit.pad_to_patch_size` | `backend.utils.pad_to_patch_size`                      |
| `comfy.samplers.SCHEDULER_NAMES`         | static list including forge-neo's scheduler names      |

The denoiser passed in by forge-neo is wrapped in a `_DenoiserAdapter`
that re-exposes the ComfyUI attribute layout (`model.inner_model
.inner_model.model_sampling`, `.device`, `.model_config`) on top of
forge-neo's `CFGDenoiserKDiffusion` / `ForgeScheduleLinker` /
`shared.sd_model` chain.

## Compatibility matrix

The samplers should work with **any** model architecture that forge-neo
itself supports - SD 1.x, SDXL, SD 3.5, Flux, Wan, etc. - because they
ride on top of `CFGDenoiser`.

### Tested against forge-neo features

These all hook in *above* the sampler layer, so they should compose
cleanly with the samplers added by this extension:

| Forge-neo feature              | Status / note                                                                                                                                                          |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--sage` (SageAttention)       | Compatible. SageAttention patches the attention layers *inside* the UNet/DiT, while the sampler only sees `CFGDenoiser`.                                               |
| `--fast-fp16` / `--fast-fp8`   | Compatible. forge-neo's `KModel.apply_model` casts inputs to the model's computation dtype, so the fp64 latents that RES4LYF uses internally are downcast safely.       |
| int8 diffusion                 | Compatible for the same reason - quantised linear layers run after `apply_model` casts inputs.                                                                          |
| torch.compile (max-autotune-no-cudagraphs) | Compatible. forge-neo compiles `KModel.apply_model` / the UNet; this sits *below* `CFGDenoiser`, so the compiled graph is still invoked through the sampler.    |
| Spectrum Integrated            | Should compose, but Spectrum's prediction caching was tuned against the stock samplers - the multistep RES samplers reuse previous-step model output, which Spectrum already caches, so quality at low step counts may differ vs. DPM++ 2M.  Disable Spectrum if you see drift. |
| LoRA / ControlNet / IP-Adapter | Compatible. They patch the UNet and are invoked via `apply_model`.                                                                                                     |
| ADM / SDXL high-res fix        | Compatible.                                                                                                                                                            |

### dtype handling

RES4LYF promotes its sampler math to `torch.float64`.  The wrapper
casts the returned latent back to whatever dtype forge-neo handed in,
so downstream code (VAE decode, hires fix, refiner, etc.) sees a
tensor in the same dtype it would get from any other sampler.

### Known limitations

- `eta` / `eta_substep` are exposed as fixed values (0.5 for the
  ancestral SDE variants, 0.0 for the ODE variants).  RES4LYF's
  per-step eta-curve and per-substep scheduling are not surfaced
  through forge-neo's UI.
- RES4LYF's `extra_options` dict, regional guides, attention masks and
  style transfer settings are also not surfaced (see "What's NOT
  included" above).

## Credits

- The samplers, schedulers and all of the math here are the work of
  [ClownsharkBatwing](https://github.com/ClownsharkBatwing) and the
  RES4LYF contributors (Apache-2.0).
- The compatibility shim and registration glue is a thin wrapper that
  doesn't reimplement any of the actual sampling math.

## Licence

`lib_res4lyf/` is Apache-2.0 (inherited from RES4LYF). See `LICENSE`.
The glue code under `scripts/` and the `_compat.py` shim are released
under the same Apache-2.0 terms.
