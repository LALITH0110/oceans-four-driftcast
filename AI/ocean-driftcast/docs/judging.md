# File Summary:
# - Provides a two-minute narration script aligned with the final competition animation.
# - Highlights key visuals, scientific beats, and calls to action for the judging panel.
# - Pair with `driftcast judge` outputs to deliver a cohesive demo.

# Driftcast Judging Script (~2 minutes)

**0:00 - 0:10 Title Card**  
Welcome and thank the panel. Introduce "Driftcast, a synthetic North Atlantic plastic drift simulator from Team Oceans Four at Illinois Tech."

**0:10 - 0:35 Scenario Setup**  
Describe the subtropical gyre domain, seeded river and shipping sources, and the physics ingredients (windage, Stokes drift, diffusion). Mention that diffusivity is scaled by latitude so mid-gyre particles move realistically in degrees.

**0:35 - 0:55 Data Integrity**  
Call out the CFL safeguard that keeps particles from tunnelling through coastlines, and highlight that every run writes a JSON manifest with git commit, config hash, and FFmpeg health checks for reproducibility.

**0:55 - 1:20 Density + Trails**  
Point to the moving density raster and fading trails. Explain how they reveal hot spots that can steer cleanup campaigns and citizen science deployments.

**1:20 - 1:40 Crowd Ingest & Analytics**  
Explain the pipeline that validates community observations (`driftcast ingest validate`), deduplicates reports, and stores parquet slices for analytics dashboards. Note the performance profiles that let laptops, workstations, and clusters trade cadence for fidelity.

**1:40 - 2:00 Call to Action**  
Close with the hero frame in the background. Summarize the one-command `driftcast judge` workflow that outputs the final MP4, hero PNG, and one-pager PDF. Invite judges to explore the docs site and preview GIF for more context, then thank them for their time.
