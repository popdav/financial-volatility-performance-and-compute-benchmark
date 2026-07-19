# SPY Dataset Inspection Report

- Requested range: ['1999-01-01', '2025-12-31']
- Actual range: ['1999-01-04', '2025-12-31']
- Observations: 6791
- Largest positive daily adjusted-price return: 0.13557734 (2008-10-13)
- Largest negative daily adjusted-price return: -0.11588651 (2020-03-16)

The configured range includes the dot-com period, global financial crisis, and
COVID-19 period. This is calendar coverage only, not algorithmic regime labeling.

## Dataset quality diagnostics

```text
                               value
trading_days                    6791
expected_trading_days_xnys      6791
missing_trading_days               0
duplicate_rows_removed             0
missing_values                     0
log_return_outliers_4_std         43
inferred_price_splits              0
inferred_dividend_adjustments    109
```

Missing XNYS sessions: none.

Expected sessions use the XNYS calendar. Outliers are inspection flags where absolute demeaned log return exceeds four sample standard deviations. Split and dividend counts are inferred from changes in the adjusted-close/close factor, with a one-basis-point noise threshold; they are not provider corporate-action records.

## First and last rows

```text
                  open        high         low       close  adjusted_close    volume
Date                                                                                
1999-01-04  123.375000  125.218750  121.718750  123.031250       76.272209   9450400
2025-12-31  687.140015  687.359985  681.710022  681.919983      678.315247  74144800
```

## Missing values

```text
                missing
open                  0
high                  0
low                   0
close                 0
adjusted_close        0
volume                0
```

## Descriptive statistics

```text
              open         high          low        close  adjusted_close        volume
count  6791.000000  6791.000000  6791.000000  6791.000000     6791.000000  6.791000e+03
mean    223.233344   224.526804   221.809339   223.249051      191.292412  1.011901e+08
std     143.206045   143.860240   142.476204   143.239666      150.875165  8.987157e+07
min      67.949997    70.000000    67.099998    68.110001       49.680573  1.436600e+06
25%     122.205002   122.965000   121.265003   122.204998       83.374332  4.757905e+07
50%     149.156250   150.139999   148.199997   149.187500      109.174698  7.570810e+07
75%     283.675003   285.364990   282.065002   283.899994      254.400131  1.283073e+08
max     690.640015   691.659973   689.270020   690.380005      686.730530  8.710263e+08
```

## Daily adjusted-price log-return statistics

```text
              value
mean       0.000322
std        0.012191
skewness  -0.202598
kurtosis  11.215627
```

## Observations by year

```text
      observations
Date              
1999           252
2000           252
2001           248
2002           252
2003           252
2004           252
2005           252
2006           251
2007           251
2008           253
2009           252
2010           252
2011           252
2012           250
2013           252
2014           252
2015           252
2016           252
2017           251
2018           251
2019           252
2020           253
2021           252
2022           251
2023           250
2024           252
2025           250
```
