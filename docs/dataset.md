# SPY Research Dataset

The primary thesis dataset is the SPDR S&P 500 ETF Trust (`SPY`) daily history.
SPY provides a long, liquid, consistently traded proxy for the broad US equity
market. The sample begins in 1999 to cover multiple major calendar periods while
avoiding the ETF's short and less representative launch history.

Returns will later be calculated from `adjusted_close`, not raw `close`, because
the adjusted series accounts for distributions and corporate actions. Dataset
preparation does not create returns, targets, or model features; returns and
21-day volatility are calculated transiently only for inspection outputs.

Prepare the dataset with:

```bash
uv run fvbench dataset prepare --config configs/research/dataset_spy.yaml
```

Use `--force-refresh` to bypass the deterministic Parquet cache. The command
creates the normalized dataset and metadata in `data/processed`, a Markdown
report and yearly summary in `results/dataset`, and three inspection plots in
`results/dataset/plots`, plus a diagram of the adjusted-price to log-return to
21-day rolling-volatility transformation flow.

The report compares observed dates with the official XNYS session calendar and
reports missing sessions. It also flags four-standard-deviation log-return
outliers. Price-split and dividend-adjustment counts are conservative inferences
from changes in the `adjusted_close / close` factor, not authoritative Yahoo
corporate-action records.

Yahoo Finance is a convenient public source, not an immutable research archive.
Historical values, adjustment calculations, API behavior, and availability can
change without versioned releases. The cache, requested dates, Git revision, and
acquisition-library versions are recorded, but a fresh retrieval is not
guaranteed to reproduce old bytes. The loader requests `auto_adjust=False` and
requires Yahoo's separate adjusted-close field; it refuses ambiguous close-only
responses rather than silently calling raw close adjusted.

Yahoo treats the request end date as exclusive in its download API. The loader
therefore requests one additional calendar day so the research configuration's
end date is inclusive. Validation still allows the final trading observation to
precede it because weekends and holidays are absent.
