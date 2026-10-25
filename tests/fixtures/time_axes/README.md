# Temporal chart fixtures

These synthetic charts were generated for this project. Each PNG has a matching JSON containing its original labels and numerical values.

- weekdays: 14 daily labels, Mon through Sun repeated twice; calendar date unknown.
- months: Jan through Dec; calendar year unknown.
- hours: 10 hourly labels from 20:00 through 05:00, crossing midnight; calendar date unknown.
- dates: 10 daily ISO labels from 2026-10-20 through 2026-10-29.

All images are 1200 × 490 pixels. Plot rectangle: left=90, top=30, right=1110, bottom=410. The y-axis spans 0–50; curve RGB is (35,110,210).

The underlying values are `22 + 5*sin(0.8*i) + 0.7*i`, with one observation at every labelled tick. These fixtures therefore support automatic sample-count suggestions; arbitrary user charts may label only some of their observations.
