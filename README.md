# Time Series Digitizer & Forecaster

A local Streamlit tool for recovering clean single-color line charts and evaluating forecasts with expanding-window rolling-origin validation.

## Run

Use Python 3.11 or newer. From this directory:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_samples.py
python -m streamlit run app.py
python -m pytest -q
```

Select a built-in sample first. For uploads, enter the plotting rectangle in image pixels, numerical axis endpoints, curve color, and the number of original equally spaced observations. Inspect the red-point overlay before forecasting. Export recovered values and backtest results as CSV.

## Architecture and methodology

`extraction.py` thresholds RGB color distance, selects the largest connected component, takes median curve rows per column, and interpolates onto a user-specified regular time grid. Linear calibration reverses the downward pixel y-axis. Coverage checks reject incomplete curves. Manual calibration is intentional: automatic OCR and axis detection are outside this initial scope.

`forecasting.py` provides damped Holt linear trend via Statsmodels and a last-value naive baseline. At each origin only the historical prefix is fitted; every complete validation fold predicts the same horizon. Errors are actual minus predicted. MAE, RMSE and failure rates are shown per lead time. The baseline also reports pooled metrics. No forecasting model selection is performed using future values.

A failure is absolute error strictly greater than the selected tolerance. The observed proportion estimates historical failure probability for each lead time. It cannot determine an exact probability for a future forecast. Overlapping folds share observations and errors are dependent; small counts and distribution shifts reduce reliability. Extraction error is not quantified by forecast backtesting. Prediction intervals are deferred until a separate calibration/coverage evaluation can be implemented.

## Supported inputs and limitations

Clean, single, continuous colored curves on linear numeric axes. Crop out legends and other curves. Black curves, disconnected traces, logarithmic axes, uncertainty bands and multiple series are unsupported. Original sampling frequency must be supplied: a raster image cannot identify it reliably. Oversampling creates artificial data. Synthetic shift charts interpolate steep segments and therefore incur raster uncertainty. The image itself stays local to the application process.

## Validation and samples

`generate_samples.py` reproducibly creates three PNG charts and original CSV data: trend, wave, and level shift. All use plot rectangle (60,30,660,330), time 0–59, values 0–100 and 60 observations. Tests compare extracted values with ground truth (MAE below one value unit), check calibration endpoints, finite Holt forecasts, naive rolling alignment, prefix invariance after changing future observations, metric calculations and invalid inputs.

## Submission

Create a GitHub repository and push this folder after reviewing it. Include sample PNG/CSV pairs. Record the demonstration using DEMO.md. Repository publication and video recording are outstanding deliverables.
