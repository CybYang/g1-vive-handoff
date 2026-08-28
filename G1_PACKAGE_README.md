# G1 Inspire G2 / VIVE handoff package

Start here:

```bash
sed -n '1,240p' docs/G1_MIGRATION_HANDOFF_ZH.md
./scripts/setup_g1_env.sh
```

This package is intentionally G1-headless. It contains VIVE UDP hand/tracker
reception, OpenXR hand processing, the current bilateral V2 DexRetargeting
pipeline, Inspire G2 URDF/mapping files, read-only checks, and guarded RS485
live launchers. It does not contain Unity, SAPIEN, camera/video, G1 arm IK, or
locomotion code.

No live launcher opens serial hardware without the exact explicit argument:

```text
RUN_LIVE_RETARGETING
```

Read the Chinese handoff document and complete its read-only checklist before
using that argument.
