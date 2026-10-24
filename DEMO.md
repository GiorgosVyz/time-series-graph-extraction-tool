# Demonstration outline (2–3 minutes)

0:00 Explain the problem and select the trend sample.
0:20 Show the automatically suggested plot rectangle and curve color. Check the axis values and original observation count. Inspect red extracted points and the extraction MAE against known sample data. Download CSV.
0:55 Select Holt, five-step horizon and a training window. Show reconstructed history and future forecast.
1:20 Explain expanding-window rolling validation: each fit sees only the historical prefix. Compare against the naive baseline at the same lead time, and open validation details to inspect the error histogram and per-lead metrics.
1:50 Change the failure tolerance. Explain that the historical proportion above tolerance estimates reliability; it is not an exact future probability. Mention overlapping folds, distribution shifts and digitization error.
2:20 Switch to the shift sample and discuss model limitations. Show the tests and README. Finish with CSV export.
