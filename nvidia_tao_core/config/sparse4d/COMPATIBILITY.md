# Sparse4D compatibility and release notes

## TAO 7.3.0 evaluation defaults

TAO 7.3.0 aligns the service schema and PyTorch CLI on `iou_3d` evaluation,
with HOTA disabled by default. TAO 7.2.0 used different defaults in these
entrypoints:

| Setting | TAO 7.2.0 Core | TAO 7.2.0 PyTorch CLI | TAO 7.3.0 Core and CLI |
| --- | --- | --- | --- |
| `dataset.eval_dist_fcn` | `iou_3d` | `center_distance` | `iou_3d` |
| `dataset.eval_hota` | `true` | `false` | `false` |

IoU3D evaluation reports `img_bbox_IoU3D/...`; center-distance evaluation uses
`img_bbox_NuScenes/...`. Explicit specifications retain their selected protocol.
To retain the TAO 7.2.0 PyTorch CLI protocol, set
`dataset.eval_dist_fcn: center_distance`. To retain the earlier Core tracking
recipe, set `dataset.eval_dist_fcn: iou_3d`, `dataset.eval_hota: true`, and include
`tracking` in `evaluate.metrics`. HOTA then reports `img_bbox_HOTA_IoU3D/...`
metrics (`img_bbox_HOTA/...` with center-distance evaluation).
Dashboards and regression comparisons must use the same protocol.

Other new co-training and numerical-recovery behavior remains opt-in.
`train.scrub_nan_gradients` also enables classification-logit clipping and
non-finite loss recovery; infinities in gradients remain visible to AMP.
`dataset.real_block_prob` accepts exactly `-1` for automatic weighting or a
probability in `[0, 1]`; fractional negative values are rejected by the runtime.

## Coordinated release gate

Ship the Core schema, PyTorch runtime, and generated skill templates together
in TAO 7.3.0. Confirm the published runtime image includes the aligned defaults
and co-training implementation before using it with service-generated specs.
Source changes alone do not update an existing container image. Release owners
manage the shared image tags for each release candidate.

## FTMS scope and artifact paths

The service schema exposes co-training options in service-generated schemas. It does not
add dataset-asset binding, upload, or path rewriting for the LTT MLP checkpoint,
LTT sidecar directory, RT-DETR cache. These values must be
explicit runtime-container-visible paths on mounted storage. Existing training,
evaluation, and inference dataset bindings remain unchanged. Automatic FTMS
binding of these auxiliary assets is a separate service integration.

Data-service `class_names` must match runtime `dataset.classes` in order.
LTT geometry uses `ltt_data/v2`, visible-box sidecars `ltt_2dgt/v1`, and
RT-DETR caches `ltt_rtdetr2d/v1`. Keep generated PKL mount paths stable.
Supervised visible-box sidecars require `instance_id` values matching the labeled
PKL's `instance_inds`, plus matching frame IDs and camera names. The producer
accepts omitted object IDs and writes `-1`, but that does not identify a labeled
object. The separate 2D-only RT-DETR cache route uses geometric matching and
does not require instance IDs.
`loss_param_touch` is DDP graph bookkeeping, not a training KPI: the runtime
omits it from logs and core does not add it to the global metric-pattern union.

Schema tests introspect dataclass defaults, UI labels, ranges, and valid options,
and call the production v1/v2 metric validators. Only intentional compatibility
choices (evaluation defaults and opt-in switches) are pinned as literal defaults.
