Heralded loss, 2% per data qubit per cycle.
| p | pairs | heralds per shot | decoder | failures X, Z | rate per cycle | mean ms | median ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| 0.003 | 1800 | 34 | kit relay-BP, heralds unread | 29, 22 | 2.3e-03 +- 3e-04 | 744 | 179 | 9533 |
| 0.003 | 1800 | 34 | kit relay-BP, heralds read | 3, 2 | 2.3e-04 +- 1e-04 | 216 | 118 | 2211 |
| 0.003 | 1800 | 34 | routed, heralds unread | 9, 7 | 7.0e-04 +- 2e-04 | 243 | 34 | 4656 |
| 0.003 | 1800 | 34 | routed, heralds read | 0, 1 | 4.6e-05 +- 5e-05 | 79 | 37 | 539 |
| 0.004 | 320 | 35 | kit relay-BP, heralds unread | 22, 25 | 1.2e-02 +- 2e-03 | 3023 | 958 | 17489 |
| 0.004 | 320 | 35 | kit relay-BP, heralds read | 2, 1 | 7.8e-04 +- 5e-04 | 649 | 213 | 7560 |
| 0.004 | 320 | 35 | routed, heralds unread | 10, 10 | 4.8e-03 +- 1e-03 | 2285 | 157 | 25641 |
| 0.004 | 320 | 35 | routed, heralds read | 1, 0 | 2.6e-04 +- 3e-04 | 284 | 47 | 5195 |
| 0.005 | 160 | 35 | kit relay-BP, heralds unread | 44, 49 | 5.3e-02 +- 6e-03 | 7991 | 6571 | 24814 |
| 0.005 | 160 | 35 | kit relay-BP, heralds read | 14, 15 | 1.4e-02 +- 3e-03 | 3074 | 859 | 18759 |
| 0.005 | 160 | 35 | routed, heralds unread | 27, 30 | 2.6e-02 +- 4e-03 | 9497 | 5752 | 46371 |
| 0.005 | 160 | 35 | routed, heralds read | 10, 5 | 6.5e-03 +- 2e-03 | 2483 | 266 | 32635 |

Share of basis-shots the herald-aware routed decoder ends in stage A: {0.003: 0.944, 0.004: 0.802, 0.005: 0.531}
zero-failure points are drawn at 2e-5; plot: results/loss_heralds.png
