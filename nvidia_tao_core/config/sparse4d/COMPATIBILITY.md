# Sparse4D compatibility and release notes

## Pending co-training release: behavior changes

This is **not a schema-only additive update**. Alongside the LTT, RT-DETR, SV2D,
and route-control fields, the shared evaluation defaults change:

| Setting | Previous core default | New shared default |
| --- | --- | --- |
| `dataset.eval_dist_fcn` | `iou_3d` | `center_distance` |
| `dataset.eval_hota` | `true` | `false` |

Distance-based evaluation reports `img_bbox_NuScenes/...`; IoU3D evaluation uses
`img_bbox_IoU3D/...`. HOTA runs only when explicitly enabled, with
`img_bbox_HOTA/...` or `img_bbox_HOTA_IoU3D/...` metrics. Dashboards and regression
comparisons must not compare these different protocols as if they were the same.

To preserve the previous core evaluation recipe, explicitly set:

```yaml
dataset:
  eval_dist_fcn: iou_3d
  eval_hota: true
```

Other new co-training and numerical-recovery behavior remains opt-in.
`train.scrub_nan_gradients` also enables classification-logit clipping and
non-finite loss recovery; infinities in gradients remain visible to AMP.
`dataset.real_block_prob` accepts exactly `-1` for automatic weighting or a
probability in `[0, 1]`; fractional negative values are rejected by the runtime.

## Coordinated release gate

Do not deploy this core schema before the runtime image actually includes
[NVIDIA-TAO/tao-pytorch#136](https://github.com/NVIDIA-TAO/tao-pytorch/pull/136).
A merged source PR alone is insufficient. The release owner must confirm the
published image tag/digest and apply the same target release label to
[core#39](https://github.com/NVIDIA-TAO/tao-core/pull/39),
[pytorch#136](https://github.com/NVIDIA-TAO/tao-pytorch/pull/136), and
[data-services#41](https://github.com/NVIDIA-TAO/tao-data-services/pull/41).
The release label/image is intentionally not guessed here.

Existing core-only fields `lazy_load`, `pkl_sample_size`, `pkl_cam_counts_path`,
`fps_drop_prob`, `target_fps_choices`, `max_cameras`, `eval_dist_fcn`, and
`eval_hota` also require the aligned runtime. This update is tested against
the matching PR source, not asserted compatible with an older runtime image.

## FTMS scope and artifact paths

This PR exposes co-training options in service-generated schemas. It does not
add dataset-asset binding, upload, or path rewriting for the LTT MLP checkpoint,
LTT sidecar directory, RT-DETR cache, or SV2D artifacts. These values must be
explicit runtime-container-visible paths on mounted storage. Existing training,
evaluation, and inference dataset bindings remain unchanged. Automatic FTMS
binding of these auxiliary assets is a separate service integration.

Data-service `class_names` must match runtime `dataset.classes` in order.
LTT geometry uses `ltt_data/v2`, visible-box sidecars `ltt_2dgt/v1`, and
RT-DETR/SV2D caches `ltt_rtdetr2d/v1`. Keep generated PKL mount paths stable.
`loss_param_touch` is DDP graph bookkeeping, not a training KPI: the runtime
omits it from logs and core does not add it to the global metric-pattern union.

Schema tests introspect dataclass defaults, UI labels, ranges, and valid options,
and call the production v1/v2 metric validators. Only intentional compatibility
choices (evaluation defaults and opt-in switches) are pinned as literal defaults.
