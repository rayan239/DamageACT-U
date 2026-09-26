# Optional qualitative xBD figure protocol

A qualitative PRE/POST/wrong-PRE figure would be useful for a computer-vision paper, but it should **not** be hand-picked after inspecting attractive examples.

If this figure is added later, freeze the following selection procedure before viewing images:

1. Use only final held-out TEST buildings for which raw xBD imagery is locally available and redistribution/display is permitted.
2. Define four semantic categories from frozen predictions **before looking at pixels**:
   - Pair helps: POST wrong, Siamese correct.
   - POST helps: POST correct, Siamese wrong.
   - Wrong-PRE harms unconditional Siamese.
   - Router-resisted corruption: Siamese changes from correct/less-wrong to harmful under wrong PRE while the routed prediction stays correct or closer to POST.
3. Stratify by held-out event where feasible.
4. Within each category/event, select the example deterministically (for example the lexicographically smallest eligible `building_id`, or a fixed seeded sample).
5. Show the target PRE, target POST, donor wrong PRE, true class, both expert predictions, router gate, and routed prediction.
6. State explicitly that the panels are illustrative and were selected by a frozen rule, not for performance estimation.

Until image-display/redistribution rights are confirmed, the quantitative paper-assets freeze is complete without this optional panel.
