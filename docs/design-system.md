# Design system

The dependency-free dashboard in `frontend/dashboard/index.html` uses semantic tokens from
`frontend/dashboard/tokens.css`. Its visual language is **Industrial Precision**: a clean light default
with a restrained blue engineering accent, a persistent day/night switch, and state colors reserved for health status.

## Tokens

- Colors: `--background`, `--surface`, `--surface-elevated`, `--border`, `--text-primary`,
  `--text-secondary`, `--text-muted`, `--accent`, `--info`, `--healthy`, `--warning`, `--degraded`,
  `--critical`.
- Spacing: `--s1`–`--s9` (4, 8, 12, 16, 24, 32, 48, 64, 96 px).
- Type: platform sans-serif for interface text; monospace for measurements and status.
- Motion: 100/150/200/300 ms tokens. Reduced-motion preferences disable nonessential animation.
- Theme preference is saved locally when browser storage is available; otherwise the current theme still works for the session.
- The PWA caches only the application shell. Prediction, health, history, and other API responses always use the network.
- Offline mode labels cached analysis as **LAST SYNCHRONIZED DATA** and shows its locally recorded sync time. Service-worker updates wait for user approval.
- Install controls appear only when the browser raises `beforeinstallprompt`. Install icons are vector SVG assets at 192 and 512 viewbox sizes.

## Application routes

The hash router supports `#/` (overview), `#/digital-twin`, `#/signal`, `#/diagnosis`, `#/health`,
`#/history`, `#/explainability`, `#/assistant`, `#/knowledge`, and `#/system`.
On narrow screens, Overview, Twin, Signal, Diagnosis, and Health remain in the bottom navigation;
History, Explainability, Assistant, Knowledge, and System are available from its accessible More menu.
The bearing schematic exposes keyboard- and pointer-selectable components. Their displayed frequency
and peak evidence comes from the latest API response, and the component link opens the corresponding
signal view. The opening and twin illustrations use upright, animated SVG schematics, not a calibrated
3D mechanical model. The overview's four bearing illustrations are explicitly not separate assets;
each opens the currently active bearing.
The overview story connects the filtered signal, returned spectrum/envelope data, physics evidence,
model output, health, and available history without synthesizing readings. History replay steps through
stored timestamp, fault, confidence, health, and alert summaries only; it does not imply archived sensor traces.
The Ctrl+K command palette contains only existing views and dashboard actions.

The UI reads prediction, spectral, physics, history, system, model, retrieval, and document-ingestion
data from the existing FastAPI endpoints. The signal view only plots the filtered waveform actually
returned by `/fft/{bearing_id}`; it does not claim the API provides raw samples. History shows only the
summary fields returned by `/history/{bearing_id}`. Document text added from the knowledge view is
indexed in the active service process, consistent with the in-memory retriever.

Synthetic sample controls remain explicit and visibly labeled as demo data. Model scores are displayed
as reported by the registry with a warning that synthetic validation is not real-machine validation.
Physics values and model explanations are never filled with UI-only example measurements.
