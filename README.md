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

The app opens on a working trend demo. Plot boundaries and the dominant saturated curve color are suggested when an image is loaded. Check the green border and red recovered points; adjust coordinates and color if necessary. Uploads require confirmation of numerical axis endpoints and the original observation count before forecasting. Example axis values are starting placeholders, not detected values. Each new image resets calibration to avoid carrying values over from another chart.

Export recovered observations, future forecasts, and rolling backtest results as CSV. Forecast controls update interactively; cached backtests avoid refitting when only the error tolerance changes.

## Architecture and methodology

`extraction.py` thresholds RGB color distance, selects the largest connected component, takes median curve rows per column, and interpolates onto a user-specified regular time grid. Linear calibration reverses the downward pixel y-axis. Coverage checks reject incomplete curves. Plot-area detection uses morphological filters to find long dark horizontal and vertical lines and requires support along all four borders. It selects the largest supported rectangle and rejects similarly sized competing areas. The result is a suggestion, not a confidence guarantee: decorative frames can also match. Charts with only two axes, faint or broken borders, or multiple panels may need manual coordinates. Numeric axis values still require manual entry; OCR is not implemented.

`forecasting.py` provides damped Holt linear trend via Statsmodels with explicit estimated initialization and a last-value naive baseline. Constant series use the constant forecast directly. At each origin only the historical prefix is fitted; every complete validation fold predicts the same horizon. Errors are actual minus predicted. Headline MAE, RMSE and failure rates compare the selected model and naive baseline at the requested terminal lead time, using identical validation origins. A detail table reports each intermediate lead time, with a histogram of terminal errors. Fewer than 20 origins triggers a limited-evidence notice; 20 is a UI heuristic, not a statistical sufficiency guarantee. No automatic model selection is performed using future values.

A failure is absolute error strictly greater than the selected tolerance. The observed proportion estimates historical failure probability for each lead time. It cannot determine an exact probability for a future forecast. Overlapping folds share observations and errors are dependent; small counts and distribution shifts reduce reliability. Extraction error is not quantified by forecast backtesting. Prediction intervals are deferred until a separate calibration/coverage evaluation can be implemented.

## Supported inputs and limitations

Clean, single, continuous colored curves on linear numeric axes. Crop out legends and other curves. Black curves, disconnected traces, logarithmic axes, uncertainty bands and multiple series are unsupported. Original sampling frequency must be supplied: a raster image cannot identify it reliably. Oversampling creates artificial data. Synthetic shift charts interpolate steep segments and therefore incur raster uncertainty. The image itself stays local to the application process.

## Validation and samples

`generate_samples.py` reproducibly creates three PNG charts and original CSV data: trend, wave, and level shift. All use plot rectangle (60,30,660,330), time 0–59, values 0–100 and 60 observations. Tests compare extracted values with ground truth (MAE below one value unit), check border/color suggestions, short and constant Holt series, rolling alignment, prefix invariance for both models after changing future observations, metric calculations and invalid inputs. The app also displays sample extraction MAE when the sample axes and observation count are used.

Streamlit interaction tests exercise startup, eight-observation inputs, model switching, invalid-calibration recovery, image switching and empty uploads. These tests need local loopback sockets. Verified locally: **28 tests passed**, on Python 3.13 with Streamlit 1.66.1 and Statsmodels 0.15.0. This is functional validation, not a guarantee for arbitrary chart styles.

## Submission

Create a GitHub repository and push this folder after reviewing it. Include sample PNG/CSV pairs. Record the demonstration using DEMO.md. Repository publication and video recording are outstanding deliverables.
