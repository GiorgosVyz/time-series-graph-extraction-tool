# Financial chart regression screenshots

These two screenshots were supplied by the user to reproduce extraction failures. They have no original numerical CSV and therefore are not used to claim full-series extraction accuracy.

- area_left_axis.png: open red shaded price chart; values on the left; month/year labels; header controls and a tooltip.
- area_right_axis.png: open green shaded price chart; grouped prices on the right; yearless month names; gray volume bars and colored point markers.

Tests check boundary geometry, saturated-stroke alignment, and rough agreement of the endpoint with the visible final price. Known-value accuracy is tested separately with synthetic charts.
