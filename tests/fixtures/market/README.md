# Financial chart regression screenshots

These screenshots were supplied by the user to reproduce extraction failures. They have no original numerical CSV and therefore are not used to claim full-series extraction accuracy.

- area_left_axis.png: open red shaded price chart; values on the left; month/year labels; header controls and a tooltip.
- area_left_axis_full.png: full CRWV screenshot with header and footer statistics, used to check automatic sampling when the original count is unknown.
- area_right_axis.png: open green shaded price chart; grouped prices on the right; yearless month names; gray volume bars and colored point markers.

- intraday_partial.png: green intraday area chart with AM/PM labels and an empty afternoon region beyond the current curve.
- month_day_month.png: red one-month chart with yearless month/day labels.
- month_day_five_days.png: red five-day chart with nonuniform month/day label spacing.

Tests check boundary geometry, saturated-stroke alignment, and rough agreement of the endpoint with the visible final price. Known-value accuracy is tested separately with synthetic charts.
