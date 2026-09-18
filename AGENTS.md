# Implementation Guardrails

`ARCHITECTURE.md` is authoritative. Read it before modifying core behavior.

Do not change without an explicit product decision:

- coordinate conventions;
- metre units;
- `UNKNOWN/FREE/OCCUPIED` meanings;
- unknown-space safety semantics;
- public model fields;
- the distinction between the diagnostic score and hard pass/fail requirements.

When adapting real engine depth:

1. Identify whether the buffer is nonlinear, linear Z-depth or radial depth.
2. Put engine conversion in an adapter, not in generic geometry functions.
3. Add a fixture captured from the real engine.
4. Add a regression test for each discovered bug.
5. Compare the occupancy debug image against known scene measurements.

Permitted improvements include vectorization, profiling, logging, service wrappers,
additional adapters, temporal fusion, rotated-rectangle search and visualization.
Keep the public API backward-compatible unless the caller approves a breaking change.

Before handoff, run:

```bash
python -m pytest
python tools/generate_synthetic_scene.py --output-dir sample_data
free-space-analyzer sample_data/depth.npy \
  --camera sample_data/camera.json \
  --config configs/default.yaml \
  --debug-png output/occupancy.png
```

