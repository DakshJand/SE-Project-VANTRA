# VANTRA Dashboard — quick guide

Five-minute walkthrough for first-time users. The dashboard runs at
`http://localhost:5173` (the API behind it at `:8712` must be running too).

## The top bar

- **VANTRA** (top left) — just the logo. Everything else lives in the search box
  and the tabs on the right panel.
- **Search box** — type a plate number (like `KA01VX2026`) and press Enter or
  click **Go**. VANTRA finds every sighting of that plate and draws its path
  across the city on the map.
- **Show heatmap** — colors the map by camera activity (more detections = hotter).
  Good for spotting which junctions are busy.
- **Operator / Analyst** — two views of the same data. Operator shows live
  monitoring tools; Analyst adds analytics and planning tools. Switch freely;
  nothing is lost.

## The map

The dark map in the middle is Bengaluru. Each dot is a camera. The faint lines
between them are the roads VANTRA knows about. When you search a plate, the
dots it passed through light up in order, with lines showing the vehicle's
route. Click a camera dot to see its details.

## The right panel tabs

**Live** — watch detections roll in as if cameras were running right now.
Click **▶ Start live replay** to replay the busiest hour of recorded traffic;
detections pop onto the map and the feed lists each one as it happens. Pause
and resume anytime. The **Video pipeline** option under this tab runs a real
traffic video through the actual detection pipeline — it's slower (~1 frame
per second) and more experimental. For a presentation, use the Demo tab
instead; use Live when you want to explore how the system behaves over time.

**Trajectory** — the main investigation view. Search a plate and this tab
fills with the vehicle's full journey: a snapshot from every camera that saw
it, and under "Hop evidence," a line for every leg of the trip explaining
exactly why VANTRA linked it — the plate read, its confidence, the travel
time. Click any row to highlight that moment on the map. There's also a
**Case file (PDF)** button that exports everything as a report.

**Alerts** — the system's automatic warnings: blacklisted plates spotted,
impossible journeys (a plate appearing in two places faster than physically
possible — the cloned-plate signal). Each alert comes with its evidence.

**Compare** — put two trajectories side by side. Useful when you suspect two
sightings are the same vehicle under different plate reads, or want to see
how a route changed day to day.

**Demo** — the reliable, presentation-ready path. Six pre-built scenarios,
one click each: a clean five-camera journey, a blurred-plate recovery, a
missed camera, a long stop, a cloned-plate alert, and a live blacklisted-plate
detection. Every one is guaranteed to work — use this tab for demos and to
show reviewers the system end-to-end.

**Audit** — the honesty check. Every match VANTRA made, scored and explained.
Use it when someone asks "why did the system link these two sightings?"

**Tuning** — adjust how strict the matching is (sliding scale between "link
everything" and "link only perfect plate reads"), with a live preview of how
your change affects results. For experimentation, not everyday use.
