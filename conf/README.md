# Attack vector set

## KPI list

| KPI                        | Index | Pearson | Minimum | Maximum | Realistic | Unrealistic | Notes             |
|----------------------------|-------|---------|---------|---------|-----------|-------------|-------------------|
| `num_ues`                  | 0     | 0.037   | 1       | 20      | 19        | 60          | 19 is not present |
| `slice_prb`                | 1     | 0.166   | 3       | 42      | 16        | 106         |                   |
| `dl_mcs`                   | 2     | -0.220  | 0       | 28      | 15        | -1          |                   |
| `dl_n_samples`             | 4     | -0.070  | 0       | 1483    | 500       | 2000        |                   |
| `dl_buffer [bytes]`        | 4     | -0.363  | 0       | 184665  | 1220      | -1          |                   |
| `tx_brate downlink [Mbps]` | 5     | -0.194  | 0       | 13.8804 | 5.88      | -2          |                   |
| `tx_pkts downlink`         | 6     | -0.069  | 0       | 571     | 450       | -3          |                   |
| `dl_cqi`                   | 7     | -0.017  | 0.1     | 15      | 8         | 50          |                   |
| `ul_mcs`                   | 8     | -0.156  | 0       | 30      | 2         | 45          |                   |
| `ul_n_samples`             | 9     | -0.070  | 0       | 1564    | 9         | -4          |                   |
| `ul_buffer [bytes]`        | 10    | -0.012  | 0       | 150460  | 84        | -5          |                   |
| `rx_brate uplink [Mbps]`   | 11    | -0.180  | 0       | 5.08237 | 0.033     | 10          |                   |
| `rx_pkts uplink`           | 12    | -0.072  | 0       | 588     | 3         | -6          |                   |
| `rx_errors uplink (%)`     | 13    | -0.009  | 0       | 100     | 15        | -7          |                   |
| `ul_sinr`                  | 14    | -0.143  | 0       | 43.0046 | 3.01      | 800         |                   |
| `phr`                      | 15    | -0.084  | 0       | 31      | 15        | 50          | Only two values   |
| `sum_requested_prbs`       | 16    | 0.150   | 0       | 11856   | 650       | 1000000     |                   |
| `sum_granted_prbs`         | 17    | -0.193  | 0       | 12528   | 2500      | -8          |                   |
| `ul_turbo_iters`           | 18    | -0.015  | 0       | 10      | 2         | -9          |                   |


## Possible target sets

1. KPI with low to no correlation with reward and realistic values
    - Pearson between -0.1 and 0.1, and uses the Realistic values
    - `one.json`
2. KPI with high correlation with reward and unrealistic values
    - Pearson > 0.15 or < -0.15, and uses unrealistic values
    - `two.json`
3. KPI with low to no correlation and unrealistic values
    - Pearson between -0.1 and 0.1, and uses “Unrealistic” values
    - `three.json`
4. KPI with high correlation with reward and realistic values
    - Pearson > 0.15 or < -0.15, and uses realistic values
    - `four.json`

