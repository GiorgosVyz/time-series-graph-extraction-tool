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

The app uses a single page: choose **Upload** for your own chart or **Demo** for Trend, Wave, or Level shift. It opens in Upload mode, with pixel-coordinate controls collapsed. Plot boundaries and the dominant saturated curve color are suggested when an image is loaded. Check the green border and contrasting recovered points; adjust coordinates and color if necessary. For uploads, review the displayed axis ranges and observation count, then click **Extract series** to continue. Changing extraction settings requires clicking the button again. Uploaded charts automatically read numeric or supported temporal x-axis labels and numeric y-axis labels using local OCR. Values are filled when the detected labels form a consistent linear scale; unreadable fields remain editable. Each new image resets calibration to avoid carrying values over from another chart.

**Download forecast** saves a self-contained interactive HTML chart containing both recovered history and future predictions. Open it in a browser, including offline. Numerical forecasts remain available via **Download forecast data (CSV)**; recovered observations and rolling backtest results also export as CSV. Forecast controls update interactively; cached backtests avoid refitting when only the error tolerance changes.

## Architecture and methodology

`extraction.py` first looks for four supported plot borders. For open charts it uses aligned, regularly spaced horizontal gridlines together with a colored curve spanning the plot. Header rules and volume regions are filtered using geometry and color evidence. Curve color is selected by horizontal coverage and saturation; tracing prefers pixels close to the saturated stroke rather than the pale shaded fill. The widest connected stroke anchors the trace, short marker gaps are interpolated, and large gaps or poor coverage are rejected. Linear calibration reverses the downward pixel y-axis. Detection remains a suggestion: complex layouts, colored volume bars, large occlusions or multiple panels may require manual correction. The local OCR calibration module is described below.

`forecasting.py` provides **ETS**, **ARIMA**, and a **Naive** last-value baseline. ETS is the default: additive errors, damped additive trend, no seasonality (ETS(A,Ad,N)), with estimated initialization. It replaces the separate Holt option. ARIMA tests only (1,0,0), (1,1,0), and (0,1,1); the stationary model includes a constant, while differenced models have no drift. At every origin, order selection uses a holdout of 2–5 observations from the end of that training prefix (one fifth of its length, within those limits). Candidates are ranked by inner-holdout MAE and refitted on the complete training prefix in that order. No outer-validation or future observations participate in order selection. ARIMA requires at least 12 training observations as a practical guard, not a guarantee of sufficient evidence. Optimizer iterations are capped, nonconverged/nonfinite candidates are rejected, and an error is reported if no candidate can be fitted. This is a restricted model comparison, not a comprehensive automatic ARIMA search. Constant series use the constant forecast directly. At each origin only the historical prefix is fitted; every complete validation fold predicts the same horizon. Errors are actual minus predicted. Headline MAE, RMSE and failure rates compare the selected model and naive baseline at the requested terminal lead time, using identical validation origins. A detail table reports each intermediate lead time, with a histogram of terminal errors. Fewer than 20 origins triggers a limited-evidence notice; 20 is a UI heuristic, not a statistical sufficiency guarantee. No automatic model selection is performed using future values.

A failure is absolute error strictly greater than the selected tolerance. The observed proportion estimates historical failure probability for each lead time. It cannot determine an exact probability for a future forecast. Overlapping folds share observations and errors are dependent; small counts and distribution shifts reduce reliability. Extraction error is not quantified by forecast backtesting. Prediction intervals are deferred until a separate calibration/coverage evaluation can be implemented.

## Automatic axis calibration

`calibration.py` uses RapidOCR and ONNX Runtime locally. Numeric words are selected by their positions along the bottom and either the left or right value axis. Conflicting left/right scales require manual selection of values. Decimal points, decimal commas, signed numbers, scientific notation, and explicit grouped prices such as `7,800.00` and `7.800,00` are supported. A comma followed by exactly three digits is treated as grouping; check ambiguous locale formats. A consensus fit requires at least three consistent ticks spanning half the axis and rejects inconsistent scales. OCR box offsets near endpoints are snapped to observed ticks or their lattice. This remains a suggestion: check the displayed values and overlay, particularly for unusual layouts. Logarithmic axes, rotated labels and unrecognized locale formats are unsupported.

First use downloads roughly 30 MB of OCR weights into `work/ocr_models/`; subsequent use runs offline. Models are not committed to Git. Only model files are downloaded; chart images are never sent to an OCR service. Install the updated requirements in the same environment used to run Streamlit. If OCR or model download fails, the app retains manual entry.

## Supported inputs and limitations

Clean, single, continuous colored curves with a linear value axis and a numeric or supported temporal x-axis. At least five original observations are required. Crop out legends and other curves. Black curves, disconnected traces, logarithmic axes, uncertainty bands and multiple series are unsupported. Original sampling frequency cannot generally be identified from a raster image. The app suggests a count from dense unit-spaced integer indices or dense regularly spaced temporal ticks covering the plot. This assumes one observation per visible label and must be checked; sparse ticks do not establish the original sample count. Oversampling creates artificial data. Synthetic shift charts interpolate steep segments and therefore incur raster uncertainty. The image itself stays local to the application process.

## Time axes and series overview

`time_axis.py` recognizes English weekdays (including the common “Wen” variant), short/full month names, 24-hour clock labels, and dates with an explicit year. ISO dates such as `2026-10-25` are preferred; named dates such as `25 Oct 2026` are also accepted. Month/year ticks such as `Jun 2026` are calibrated at the start of the month; intermediate endpoints are approximate and marked as estimated. The time-axis selector and start/end fields allow correction. Numeric-axis behavior is preserved.

Weekdays and month names retain their labels without inventing a calendar year. Clock, weekday, and month rollovers are unwrapped only when at least three labels fit a consistent linear progression. Historical midnight or cycle crossings are marked, for example `00:30 (+1 day)`, `Mon (+1 week)`, and `Jan (+1 year)`. Calendar dates retain their explicit year. Ambiguous numeric dates (`03/04`), yearless calendar dates (`May 12`), irregular calendars, timezone changes, and business-day-only spacing are not inferred automatically.

**Observation frequency** distinguishes **Unknown**, **Daily**, and **Intraday** sampling. Sparse month ticks default to Unknown; select Daily only when the original points are known to represent days. Visible clock labels suggest Intraday without assuming an hourly interval. Calendar labels show dates only unless Intraday is selected and actual time labels are available. Hours and minutes produced by interpolating month/year ticks are hidden, including in the overview, hover labels, and CSVs. Changing frequency requires extracting again, and each new image resets it. Selecting Daily changes presentation; it does not reconstruct missing trading timestamps or aggregate the recovered values.

Temporal forecasts use `Observation +1`, etc., or `Daily observation +1` for confirmed daily sampling. These labels are consistent in chart ticks, hover text, HTML and CSV downloads. Future positions are drawn with regular spacing for readability; exact future calendar dates, intraday timestamps, exchange sessions and holidays are not inferred. Numeric-axis forecasts continue their numeric spacing.

Recognized labels appear in chart ticks, hover values, the interactive HTML export, and recovered/forecast CSVs. Backtest CSVs include forecast-origin and target labels. The values used to fit forecasting models remain the recovered numeric observations.

The **Series overview** shows the observation count and readable period, followed by **Estimated high**, **Estimated low**, and **Overall change**. Approximate calendar periods are marked, without generated hours and minutes. A note about first occurrences appears only for repeated extrema. Constant series are described without claiming a trend. It is generated directly from extracted values, uses no language-model API and makes no causal or statistical-significance claims. Dense tick labels may still represent fewer points than the original series; check the suggested observation count before interpreting the overview.

Four additional upload examples and their known values are in `tests/fixtures/time_axes/`: weekdays, months, hours crossing midnight, and ISO dates. All four were verified with the real local OCR engine through upload, extraction, overview and forecasting.

## Simple financial area charts

Open charts with a single saturated price line, pale area fill, left/right numeric scales, and sparse month labels are supported. Gray volume bars, current-price badges, and small point markers are excluded where possible. The preview uses blue recovered points on a red curve, and red points on other curves, for visibility. A trace preview is shown while the original sample count is awaiting confirmation.

Sparse month labels do not determine the original sample count, trading calendar, or exact endpoints. Enter the known count to proceed; the application does not invent a count from image width. Values are resampled at regular chart positions, which cannot recover missing source trading timestamps or irregular sampling. Yearless month labels retain relative month positions (including editable fractional-month offsets); month/year labels provide an approximate calendar period. Candlesticks, multiple price curves, logarithmic scales and substantial occlusions remain outside scope.

The two user-supplied screenshots in `tests/fixtures/market/` are regression examples, not datasets with complete source ground truth. Real OCR and the full upload-to-forecast flow were checked on both, using 40 explicitly specified samples for the functional test only. Endpoint checks were close to the visible price labels; full-series accuracy is not claimed. A separate synthetic shaded chart with known values tests extraction MAE below 1.5 units, including headers and gray volume bars.

## Validation and samples

`generate_samples.py` reproducibly creates three PNG charts and original CSV data: trend, wave, and level shift. All use plot rectangle (60,30,660,330), time 0–59, values 0–100 and 60 observations. Tests compare extracted values with ground truth (MAE below one value unit), check border/color suggestions, short and constant ETS series, ARIMA minimum training size and fitting failures, rolling alignment, prefix invariance for ETS, ARIMA and Naive after changing future observations, metric calculations and invalid inputs. The app also displays sample extraction MAE when the sample axes and observation count are used.

Streamlit interaction tests exercise startup, eight-observation inputs, model switching, invalid-calibration recovery, image switching, empty uploads, upload confirmation and changes to extraction settings. Temporal checks cover daily date precision, approximate periods, intraday source evidence, forecast labels in charts and CSVs, and frequency changes/reset. These tests need local loopback sockets. Verified locally: **82 tests passed**, on Python 3.13 with Streamlit 1.66.1 and Statsmodels 0.15.0. This is functional validation, not a guarantee for arbitrary chart styles.

## Submission

Create a GitHub repository and push this folder after reviewing it. Include sample PNG/CSV pairs. Record the demonstration using DEMO.md. Repository publication and video recording are outstanding deliverables.
