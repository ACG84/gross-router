| p | pairs | decoder | failures X, Z | rate per cycle | mean ms | median ms | p99 ms |
|---|---|---|---|---|---|---|---|
| 0.003 | 7000 | kit relay-BP | 2, 4 | 7.1e-05 +- 3e-05 | 88 | 54 | 583 |
| 0.003 | 7000 | routed, hard side answer | 0, 2 | 2.4e-05 +- 2e-05 | 17 | 10 | 130 |
| 0.003 | 7000 | routed, soft side beliefs (raw BP) | 0, 1 | 1.2e-05 +- 1e-05 | 15 | 10 | 85 |
| 0.003 | 7000 | routed, learned calibration | 0, 1 | 1.2e-05 +- 1e-05 | 15 | 10 | 87 |
| 0.004 | 2000 | kit relay-BP | 12, 6 | 7.5e-04 +- 2e-04 | 311 | 102 | 5311 |
| 0.004 | 2000 | routed, hard side answer | 2, 2 | 1.7e-04 +- 8e-05 | 124 | 16 | 3075 |
| 0.004 | 2000 | routed, soft side beliefs (raw BP) | 2, 1 | 1.3e-04 +- 7e-05 | 53 | 16 | 419 |
| 0.004 | 2000 | routed, learned calibration | 2, 2 | 1.7e-04 +- 8e-05 | 46 | 16 | 351 |
| 0.005 | 560 | kit relay-BP | 23, 23 | 7.0e-03 +- 1e-03 | 1520 | 408 | 10507 |
| 0.005 | 560 | routed, hard side answer | 5, 9 | 2.1e-03 +- 6e-04 | 1201 | 71 | 16420 |
| 0.005 | 560 | routed, soft side beliefs (raw BP) | 3, 9 | 1.8e-03 +- 5e-04 | 310 | 62 | 7692 |
| 0.005 | 560 | routed, learned calibration | 2, 7 | 1.3e-03 +- 5e-04 | 215 | 62 | 3398 |

Shots that stage A does not settle (the only ones the side information touches):
| p | share of basis-shots | variant | failures | mean ms | p99 ms |
|---|---|---|---|---|---|
| 0.003 | 3.5% | routed, hard side answer | 1 | 75 | 888 |
| 0.003 | 3.5% | routed, soft side beliefs (raw BP) | 0 | 38 | 336 |
| 0.003 | 3.5% | routed, learned calibration | 0 | 40 | 365 |
| 0.004 | 13.7% | routed, hard side answer | 3 | 378 | 5338 |
| 0.004 | 13.7% | routed, soft side beliefs (raw BP) | 2 | 122 | 1271 |
| 0.004 | 13.7% | routed, learned calibration | 3 | 94 | 1147 |
| 0.005 | 38.1% | routed, hard side answer | 13 | 1532 | 10808 |
| 0.005 | 38.1% | routed, soft side beliefs (raw BP) | 11 | 364 | 8785 |
| 0.005 | 38.1% | routed, learned calibration | 8 | 240 | 4129 |

zero-failure points are drawn at 5e-6; plot: results/learned_calibration.png
