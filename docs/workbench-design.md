# Workbench presentation

OPEN JENSEN uses a shared visual system for the workspace and API reference: cool neutral surfaces, a restrained blue selection color, readable stage headings, consistent control spacing, and matching light/dark tokens. Focus indicators, reduced-motion preferences, existing accessible control names, and narrow-screen wrapping remain part of the shell. On narrow screens, the grouped Data, Train, Test and Workspace navigation forms a compact horizontally scrollable strip, with every destination keyboard reachable; project selection and creation share a compact row. Blue indicates selection; semantic green remains reserved for recorded success or connected status. Decoration does not imply connectivity or model quality.

The sidebar groups work by purpose:

- **Data:** Dataset, Augmentation and Teaching.
- **Train:** Fine-tune, Distill and Quantize.
- **Test:** Evaluate, Run and Decision lab.
- **Workspace:** Cloud runs and Settings & diagnostics.

Augmentation, Teaching and Decision lab have their own destinations. Dataset keeps its Sources and Inspection views, with an **Augment this dataset** handoff from an inspection. Each destination uses one page title. Explanatory subtitles and repeated introductions are omitted; section labels organize the actual controls and results. The [workspace guide](workspace-guide.md), available through **Guide** at `/guide/` on desktop and mobile, explains what each sector does and how to use it. `/docs/` remains the API reference.

Run presents three visible choice cards: **3D simulation**, **Replay observations** and **Check inference**. Quantize presents text-only **SmolVLA** and **ACT** cards with their supported formats; decorative symbols do not stand in for model logos. Policy, compute and method choices use labeled cards where appropriate. Keep labels short; show compatibility, availability and required constraints where they affect a choice. Move general explanations and learning material to the guide; do not add unsupported-family subtitles below choice cards. Generator, control source and advisory model are identified with compact scoped cards inside their generic workflow pages. Distill opens a Supported models overview with no default family. ACT (Action Chunking with Transformers) is one explicitly selected adapter, currently the only supported family; saved jobs and request recovery remain reachable from the overview. Fresh Fine-tune, Quantize and Run workflows also require an explicit model or execution-mode choice. Saved user preferences, owned job history, resume requests and unresolved submissions retain their exact context. Selecting a mode opens its workflow without submitting a job. Saved results remain readable when their worker is unavailable.

Augmentation follows source clips, appearance, then generation and review. Dataset and camera cards, episode quick picks, illustrated appearance choices and visible result selectors replace the former selection menus. Empty states offer dataset import or provider setup at the point where they are needed. Billing disclosure, validation, cancellation and source provenance remain available.

These presentation changes preserve the distinction between engine execution checks, scored evaluation, native simulation, observation replay and download-only packed policies. Inference checks establish loading and finite-action behavior, not robot task success. Offline replay does not establish closed-loop success; experimental Isaac recordings retain their unmeasured pickup-success and unverified-calibration labels.

Native simulation separates reviewing work from preparing another paid run. Selecting a recorded job brings its status and video ahead of the preparation form. Pickup success and calibration stay explicitly unmeasured/unverified. Known worker states remain visible while activity, identity, and raw reports live in an expandable technical section. Polling preserves the user's disclosure choice. Failed activity refreshes leave a visible stale-state notice alongside cached worker observations, even when technical details are closed. The new-run form remains reachable by keyboard and retains source/profile selection, timeout bounds, ambiguous-submission handling, and explicit paid-run consent; no launch occurs on navigation.

Browser regression suites cover desktop, mobile, API and real subprocess workflows, including exact requests, clip selection limits, result comparison and downloads, cancellation, recovery after uncertain submissions, project isolation, Distill → Quantize → Replay handoffs, keyboard use, theme switching and 320px overflow. Visual review uses the production export on a temporary local workspace. Generated API/media fixtures used for layout and browser checks are not robotics evidence or proof of live paid-provider access.


Editable forms use explicit scoped layout: labels above full-width controls,
borderless fieldsets, visible spacing between fields, and responsive action rows.
Decision and Teaching share these rules; checkbox consent remains on its own row.
Layout checks measure label/control bounds and widths, not only visibility or
page overflow. Default comparison instructions must fit without being clipped.
