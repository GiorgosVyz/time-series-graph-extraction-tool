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

The app uses a single page: choose **Upload** for your own chart or **Demo** for Trend, Wave, or Level shift. It opens in Upload mode, with pixel-coordinate controls collapsed. Plot boundaries and the dominant saturated curve color are suggested when an image is loaded. Check the green border and red recovered points; adjust coordinates and color if necessary. For uploads, review the displayed axis ranges and observation count, then click **Extract series** to continue. Changing extraction settings requires clicking the button again. Uploaded charts automatically read numeric or supported temporal x-axis labels and numeric y-axis labels using local OCR. Values are filled when the detected labels form a consistent linear scale; unreadable fields remain editable. Each new image resets calibration to avoid carrying values over from another chart.

**Download forecast** saves a self-contained interactive HTML chart containing both recovered history and future predictions. Open it in a browser, including offline. Numerical forecasts remain available via **Download forecast data (CSV)**; recovered observations and rolling backtest results also export as CSV. Forecast controls update interactively; cached backtests avoid refitting when only the error tolerance changes.

## Architecture and methodology

`extraction.py` thresholds RGB color distance, selects the largest connected component, takes median curve rows per column, and interpolates onto a user-specified regular time grid. Linear calibration reverses the downward pixel y-axis. Coverage checks reject incomplete curves. Plot-area detection uses morphological filters to find long dark horizontal and vertical lines and requires support along all four borders. It selects the largest supported rectangle and rejects similarly sized competing areas. The result is a suggestion, not a confidence guarantee: decorative frames can also match. Charts with only two axes, faint or broken borders, or multiple panels may need manual coordinates. The local OCR calibration module is described below.

`forecasting.py` provides **ETS**, **ARIMA**, and a **Naive** last-value baseline. ETS is the default: additive errors, damped additive trend, no seasonality (ETS(A,Ad,N)), with estimated initialization. It replaces the separate Holt option. ARIMA tests only (1,0,0), (1,1,0), and (0,1,1); the stationary model includes a constant, while differenced models have no drift. At every origin, order selection uses a holdout of 2–5 observations from the end of that training prefix (one fifth of its length, within those limits). Candidates are ranked by inner-holdout MAE and refitted on the complete training prefix in that order. No outer-validation or future observations participate in order selection. ARIMA requires at least 12 training observations as a practical guard, not a guarantee of sufficient evidence. Optimizer iterations are capped, nonconverged/nonfinite candidates are rejected, and an error is reported if no candidate can be fitted. This is a restricted model comparison, not a comprehensive automatic ARIMA search. Constant series use the constant forecast directly. At each origin only the historical prefix is fitted; every complete validation fold predicts the same horizon. Errors are actual minus predicted. Headline MAE, RMSE and failure rates compare the selected model and naive baseline at the requested terminal lead time, using identical validation origins. A detail table reports each intermediate lead time, with a histogram of terminal errors. Fewer than 20 origins triggers a limited-evidence notice; 20 is a UI heuristic, not a statistical sufficiency guarantee. No automatic model selection is performed using future values.

A failure is absolute error strictly greater than the selected tolerance. The observed proportion estimates historical failure probability for each lead time. It cannot determine an exact probability for a future forecast. Overlapping folds share observations and errors are dependent; small counts and distribution shifts reduce reliability. Extraction error is not quantified by forecast backtesting. Prediction intervals are deferred until a separate calibration/coverage evaluation can be implemented.

## Automatic axis calibration

`calibration.py` uses RapidOCR and ONNX Runtime locally. Numeric words are selected by their positions along the bottom and left axes. Decimal points, decimal commas, signed numbers and scientific notation are supported. A consensus fit requires at least three consistent ticks spanning half the axis and rejects inconsistent scales. OCR box offsets near endpoints are snapped to observed ticks or their lattice. This remains a suggestion: check the displayed values and overlay, particularly for unusual layouts. Logarithmic axes, rotated labels and ambiguous thousands separators are unsupported.

First use downloads roughly 30 MB of OCR weights into `work/ocr_models/`; subsequent use runs offline. Models are not committed to Git. Only model files are downloaded; chart images are never sent to an OCR service. Install the updated requirements in the same environment used to run Streamlit. If OCR or model download fails, the app retains manual entry.

## Supported inputs and limitations

Clean, single, continuous colored curves with a linear value axis and a numeric or supported temporal x-axis. At least five original observations are required. Crop out legends and other curves. Black curves, disconnected traces, logarithmic axes, uncertainty bands and multiple series are unsupported. Original sampling frequency cannot generally be identified from a raster image. The app suggests a count from dense unit-spaced integer indices or dense regularly spaced temporal ticks covering the plot. This assumes one observation per visible label and must be checked; sparse ticks do not establish the original sample count. Oversampling creates artificial data. Synthetic shift charts interpolate steep segments and therefore incur raster uncertainty. The image itself stays local to the application process.

## Time axes and series overview

`time_axis.py` recognizes English weekdays (including the common “Wen” variant), short/full month names, 24-hour clock labels, and dates with an explicit year. ISO dates such as `2026-10-25` are preferred; named dates such as `25 Oct 2026` are also accepted. The time-axis selector and start/end fields allow correction. Numeric-axis behavior is preserved.

Weekdays and month names retain their labels without inventing a calendar year. Clock, weekday, and month rollovers are unwrapped only when at least three labels fit a consistent linear progression. Forecast labels continue the same regular interval. Midnight or cycle crossings are marked, for example `00:30 (+1 day)`, `Mon (+1 week)`, and `Jan (+1 year)`. Calendar dates retain their explicit year. Ambiguous numeric dates (`03/04`), yearless calendar dates (`May 12`), irregular calendars, timezone changes, and business-day-only spacing are not inferred automatically.

Recognized labels appear in chart ticks, hover values, the interactive HTML export, and recovered/forecast CSVs. Backtest CSVs include forecast-origin and target labels. The values used to fit forecasting models remain the recovered numeric observations.

The **Series overview** is generated directly from extracted values: observation count, recognized period, estimated extrema and their first occurrence, and change from the first to last value. Constant series are described without claiming a trend. It uses no language-model API and makes no causal or statistical-significance claims. Dense tick labels may still represent fewer points than the original series; check the suggested observation count before interpreting the overview.

Four additional upload examples and their known values are in `tests/fixtures/time_axes/`: weekdays, months, hours crossing midnight, and ISO dates. All four were verified with the real local OCR engine through upload, extraction, overview and forecast-label continuation.

## Validation and samples

`generate_samples.py` reproducibly creates three PNG charts and original CSV data: trend, wave, and level shift. All use plot rectangle (60,30,660,330), time 0–59, values 0–100 and 60 observations. Tests compare extracted values with ground truth (MAE below one value unit), check border/color suggestions, short and constant ETS series, ARIMA minimum training size and fitting failures, rolling alignment, prefix invariance for ETS, ARIMA and Naive after changing future observations, metric calculations and invalid inputs. The app also displays sample extraction MAE when the sample axes and observation count are used.

Streamlit interaction tests exercise startup, eight-observation inputs, model switching, invalid-calibration recovery, image switching, empty uploads, upload confirmation and changes to extraction settings. These tests need local loopback sockets. Verified locally: **63 tests passed**, on Python 3.13 with Streamlit 1.66.1 and Statsmodels 0.15.0. This is functional validation, not a guarantee for arbitrary chart styles.

## Submission

Create a GitHub repository and push this folder after reviewing it. Include sample PNG/CSV pairs. Record the demonstration using DEMO.md. Repository publication and video recording are outstanding deliverables.
