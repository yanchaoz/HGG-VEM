# Release boundary

- [x] Four requested core code modules selected.
- [x] U-Net entry selected for masked image modeling.
- [x] Contact grid distinguished from public image sampling.
- [x] Mesh IDs aligned to 2676 / 530 / 624 without changing source files.
- [x] Formal morphometric measurement code excluded.
- [ ] Full MIM / point-cloud GPU training and checkpoint-to-prediction replay (not performed during packaging).
- [x] Preserve retrieved upstream licence texts and source notices; no new blanket project licence assigned.
- [ ] Maintainers: complete project-level licence/provenance review and confirm human-tissue sharing authorization for the separate dataset.
- [ ] Upload the separate dataset and test downloads from a reviewer-equivalent account; the code PR does not perform this step.
- [ ] Narrow manuscript/response availability wording: current release excludes morphometric measurement code, trained weights, reference annotations, and ER/Golgi masks. No manuscript was edited during packaging.

See `TEST_REPORT.json` for actual completed tests; a syntax check is not a training reproduction. `internal/` belongs to the parent staging directory and must not be uploaded as part of either public repository.
