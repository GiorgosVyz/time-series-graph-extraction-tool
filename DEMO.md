# Demonstration video

The video is provided separately as an attachment to the submission email.

**File:** ODYCEO_Demo_Final_v2.mp4

**Duration:** 2:12 · **Resolution:** 1920 × 1080 · **Language:** English on-screen captions

The walkthrough uses the uploaded [CRWV chart](tests/fixtures/market/area_left_axis_full.png).

| Time | What is shown |
| --- | --- |
| 0:00–0:36 | Upload the image, review the detected axes and sampling settings, and inspect the plot boundary and extracted points. |
| 0:36–0:55 | Review the recovered series and its overview. |
| 0:55–1:37 | Show the forecasting controls, ETS forecast and historical accuracy metrics against the Naive baseline. |
| 1:37–1:56 | Inspect the forecast error distribution: percentage errors, tolerance boundaries, overestimates and underestimates. |
| 1:56–2:02 | Compare MAE, RMSE and historical failure rate at each lead time. |
| 2:02–2:12 | Download the interactive HTML forecast and open it offline. |

The example uses 181 interpolated calendar-day samples, a five-step horizon and ±10% tolerance. Its backtest evaluates the sampled curve. Historical failure rate is not an exact probability of future error.
