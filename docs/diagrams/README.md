# Diagrams

This folder contains pipeline diagrams in multiple formats.

## Source files (`mmd/`)

The `.mmd` files in `docs/diagrams/mmd/` are the **editable Mermaid source**. These are plain text and version-controlled. Edit these to update diagrams.

| File | Description |
|------|-------------|
| `core_pipeline_map.mmd` | Core data processing pipeline (processing scripts 01–07) |
| `pipeline_flowchart.mmd` | High-level pipeline flowchart for README/paper reference |

## Exported files (`svg/`, `png/`)

- `svg/` — vector exports for embedding in docs, web, LaTeX
- `png/` — raster exports for README previews and presentations

SVG and PNG exports have been generated from the `.mmd` source files using `@mermaid-js/mermaid-cli` (installed locally as a dev dependency):

| Source | SVG | PNG |
|--------|-----|-----|
| `mmd/core_pipeline_map.mmd` | `svg/core_pipeline_map.svg` | `png/core_pipeline_map.png` |
| `mmd/pipeline_flowchart.mmd` | `svg/pipeline_flowchart.svg` | `png/pipeline_flowchart.png` |

## How to regenerate exports

Mermaid CLI is installed locally as a dev dependency (`package.json` in repo root).

```bash
# Install (if node_modules not present):
npm install

# Regenerate all exports:
npx mmdc -i docs/diagrams/mmd/core_pipeline_map.mmd -o docs/diagrams/svg/core_pipeline_map.svg
npx mmdc -i docs/diagrams/mmd/core_pipeline_map.mmd -o docs/diagrams/png/core_pipeline_map.png
npx mmdc -i docs/diagrams/mmd/pipeline_flowchart.mmd -o docs/diagrams/svg/pipeline_flowchart.svg
npx mmdc -i docs/diagrams/mmd/pipeline_flowchart.mmd -o docs/diagrams/png/pipeline_flowchart.png
```

## Manual alternatives (no Node.js required)

### Option A — Online Mermaid editor

Paste the contents of any `.mmd` file into https://mermaid.live and export as SVG or PNG.

### Option B — VS Code extension

Install the **Mermaid Preview** extension in VS Code to render diagrams inline.
