# Roadmap

This roadmap describes the near-, mid-, and long-term direction for
**frontline-drones**. It is a *reference catalog* first and a small query toolkit
second; every item below preserves that scope — a descriptive, citation-grade
index of what publicly exists, with sources, and **no** operational, flight,
guidance, or targeting content (see [DISCLAIMER.md](DISCLAIMER.md)).

Nothing here is a commitment or a schedule; it is a statement of intended
direction and an invitation for contributions.

## Guiding principles

- **Citation-grade or it doesn't ship.** Every row keeps a primary source; every
  spec is a *publicly reported* figure, annotated when sources disagree.
- **Additive and backward-compatible.** Existing CSV schemas, the validator, the
  installer, and the `frontline-drones` CLI stay stable. New columns and
  subcommands are added; existing ones are not broken or removed.
- **Dependency-light.** The runtime query engine and tools stay standard-library
  only. Heavy dependencies (ML runtimes) remain opt-in via `install_models.py`.
- **Defensive framing.** Counter-UAS and detection guidance stays sensor- and
  selection-oriented, never operational.

## Near-term (next few releases)

- **Catalog breadth.** Expand each dataset with additional well-sourced rows
  (more ISR platforms, more open-autonomy stacks, more NVIDIA model families) and
  keep the per-row primary source discipline.
- **Schema enrichment (additive columns).** Optional columns such as
  `first_seen_year`, `status`, and `region` on the military set; `country` and
  `open_source` flags on the commercial set — added without breaking the
  validator's required-column contract.
- **CLI ergonomics.** `--sort` with secondary keys, `--distinct` output, and a
  `--count` summary mode. A `frontline-drones validate` subcommand that wraps the
  existing `scripts/validate.py` checks.
- **More output formats.** NDJSON (one finding per line) for streaming into
  pipelines, and a compact GeoJSON export where a row carries a location.

## Mid-term

- **Cross-dataset views.** A unified query surface (`--dataset all`) that searches
  every catalog at once and tags results by dataset.
- **Source health.** An opt-in link-checker (using the bundled `livesearch`
  fetch primitives) that flags dead or moved `source_url`/`docs_url`/`url`
  entries in CI as a warning, never a hard failure.
- **Provenance metadata.** A machine-readable `sources.yml` mapping every row to
  one or more citations with retrieval dates, powering a generated `SOURCES.md`.
- **Packaged data.** Ship the CSVs inside the wheel so `pip install
  frontline-drones` works without a checkout, with the query engine resolving
  bundled data automatically.

## Long-term

- **Integration surface.** First-class, documented JSON/finding output that maps
  cleanly through `cognis-connect` (see [INTEGRATIONS.md](INTEGRATIONS.md)) into
  STIX/MISP and analyst-brief destinations.
- **Automated freshness.** A scheduled, review-gated refresh that proposes catalog
  additions from public reporting for human verification — never auto-merged.
- **Interoperability.** Stable identifiers and a documented schema versioning
  policy so downstream tools can depend on the catalog as a data contract.

## Explicitly out of scope

- Any operational, assembly, flight-control, navigation, guidance, or targeting
  content, or instructions to weaponize any platform.
- Non-public or restricted specifications. Only openly reported figures with a
  citation are eligible.

Contributions that add well-sourced rows, tighten citations, or extend the query
tooling within these principles are welcome.
