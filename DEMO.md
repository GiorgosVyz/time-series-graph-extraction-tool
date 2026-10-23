# Demonstration outline (2–3 minutes)

0:00 Explain the problem and select the trend sample.
0:20 Show the plot rectangle, linear axis values, original observation count and curve color. Inspect red extracted points and download CSV.
0:55 Select Holt, five-step horizon and a training window. Show reconstructed history and future forecast.
1:20 Explain expanding-window rolling validation: each fit sees only the historical prefix. Compare against the naive baseline and inspect errors by lead time.
1:50 Change the failure tolerance. Explain that the historical proportion above tolerance estimates reliability; it is not an exact future probability. Mention overlapping folds, distribution shifts and digitization error.
2:20 Switch to the shift sample and discuss model limitations. Show the tests and README. Finish with CSV export.
