# Feature Engineering and Supervised Dataset

All features at timestamp \(t\) use observations dated no later than \(t\). Prices
for returns, trend, and momentum are the adjusted close \(P_t\); ATR deliberately
uses the raw high, low, and close because it measures the traded daily range. No
missing observation is forward-filled.

## Definitions

- **Log return (price):** \(r_t=\ln(P_t/P_{t-1})\). The first value is missing.
- **Simple return (price):** \(P_t/P_{t-1}-1\).
- **Historical volatility (volatility):** for \(w\in\{5,10,21,63\}\), the
  sample standard deviation (denominator \(w-1\)) of \(r_{t-w+1},\ldots,r_t\),
  multiplied by \(\sqrt{252}\). It measures backward-looking variability.
- **SMA (trend):** for \(w\in\{5,10,21,50\}\),
  \(w^{-1}\sum_{i=0}^{w-1}P_{t-i}\).
- **EMA (trend):** for \(w\in\{10,21,50\}\),
  \(EMA_t=\alpha P_t+(1-\alpha)EMA_{t-1}\), where \(\alpha=2/(w+1)\).
  Output begins after a complete \(w\)-observation warm-up.
- **RSI(14) (momentum):** price changes are separated into gains and losses.
  Their initial 14-day arithmetic averages seed Wilder recursions
  \(A_t=((13)A_{t-1}+x_t)/14\). Then \(RS=A^+_t/A^-_t\) and
  \(RSI=100-100/(1+RS)\). A gain-only window maps to 100; a flat window to 50.
- **ATR(14) (volatility):** true range is the maximum of high minus low,
  \(|high_t-close_{t-1}|\), and \(|low_t-close_{t-1}|\). ATR is its 14-day
  Wilder average.
- **Log volume (volume):** \(\ln(volume_t)\).
- **Volume change (volume):** \(volume_t/volume_{t-1}-1\).

The rolling statistics expose recent scale, trend, momentum, range, and trading
activity so statistical, tabular, and sequence models receive one common input
definition. YAML controls every group and window; disabled groups produce no
columns.

## Forecast Target

For horizon \(h\), only \(h=5\) and \(h=21\) are supported:

\[
RV(t,h)=\sqrt{\frac{1}{h}\sum_{i=1}^{h}r_{t+i}^{2}}
         \sqrt{\frac{252}{h}}.
\]

Thus a feature row dated \(t\) predicts adjusted-price returns dated \(t+1\)
through \(t+h\). Neither \(r_t\) nor an earlier return enters the target. The
final \(h\) rows are missing because a complete future window is unavailable.
For example, a five-day row on 2026-01-05 uses returns dated 2026-01-06 through
2026-01-10 and is scaled by \(\sqrt{252/5}\).

Features and target are joined on their `DatetimeIndex` first. Only then are
warm-up rows, incomplete future targets, NaNs, and infinities removed. Validation
rejects duplicate indices or names, misalignment, non-finite values, and constant
features, and reports the input, output, and removed row counts.
