# E1 calibration report

| run_id | workload | run status | channel | health | samples | missing | align ms | edge ms |
|---|---|---|---|---|---|---|---|---|
| 20260824_calib-bursty_node2-gpu0_s0_r01 | calib_bursty | completed | nvml.gpu0 | pass | 256 | 0.0 | 90.2 | 2379.0 |
| 20260824_calib-bursty_node2-gpu1_s0_r01 | calib_bursty | flagged_channel_health | nvml.gpu1 | fail | 254 | 0.0 | 134.2 | 2119.0 |
| 20260824_calib-bursty_node2-gpu1_s0_r02 | calib_bursty | completed | nvml.gpu1 | pass | 254 | 0.0 | 43.3 | 2120.1 |
| 20260824_calib-bursty_node2-gpu2_s0_r01 | calib_bursty | flagged_channel_health | nvml.gpu2 | fail | 258 | 0.0 | 167.2 | 2099.4 |
| 20260824_calib-bursty_node2-gpu2_s0_r02 | calib_bursty | completed | nvml.gpu2 | pass | 258 | 0.0 | 65.3 | 2107.9 |
| 20260824_calib-bursty_node2-gpu3_s0_r01 | calib_bursty | completed | nvml.gpu3 | pass | 272 | 0.0 | 89.4 | 2229.3 |
| 20260824_calib-bursty_node2-gpu4_s0_r01 | calib_bursty | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 108.9 | None |
| 20260824_calib-bursty_node2-gpu4_s0_r02 | calib_bursty | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 38.4 | None |
| 20260824_calib-gemm_node2-gpu0_s0_r01 | calib_gemm | completed | nvml.gpu0 | pass | 238 | 0.0 | 52.0 | 2177.4 |
| 20260824_calib-gemm_node2-gpu1_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu1 | fail | 236 | 0.0 | 134.5 | 2185.6 |
| 20260824_calib-gemm_node2-gpu1_s0_r02 | calib_gemm | completed | nvml.gpu1 | pass | 236 | 0.0 | 57.0 | 1962.5 |
| 20260824_calib-gemm_node2-gpu2_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu2 | fail | 240 | 0.0 | 168.0 | 1634.3 |
| 20260824_calib-gemm_node2-gpu2_s0_r02 | calib_gemm | completed | nvml.gpu2 | pass | 240 | 0.0 | 92.1 | 1670.9 |
| 20260824_calib-gemm_node2-gpu3_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu3 | fail | 248 | 0.0 | 110.1 | 2145.9 |
| 20260824_calib-gemm_node2-gpu3_s0_r02 | calib_gemm | completed | nvml.gpu3 | pass | 248 | 0.0 | 62.2 | 2144.2 |
| 20260824_calib-gemm_node2-gpu4_s0_r01 | calib_gemm | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 308 | 0.0 | 102.7 | None |
| 20260824_calib-gemm_node2-gpu4_s0_r02 | calib_gemm | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 296 | 0.0 | 38.5 | None |
| 20260824_calib-idle_node2-gpu0_s0_r01 | calib_idle | completed | nvml.gpu0 | pass | 235 | 0.0 | 75.7 | 2250.2 |
| 20260824_calib-idle_node2-gpu1_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu1 | fail | 234 | 0.0 | 135.3 | 2050.2 |
| 20260824_calib-idle_node2-gpu1_s0_r02 | calib_idle | completed | nvml.gpu1 | pass | 234 | 0.0 | 31.2 | 2199.3 |
| 20260824_calib-idle_node2-gpu2_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu2 | fail | 238 | 0.0 | 167.4 | 2125.9 |
| 20260824_calib-idle_node2-gpu2_s0_r02 | calib_idle | completed | nvml.gpu2 | pass | 237 | 0.0 | 47.1 | 2176.0 |
| 20260824_calib-idle_node2-gpu3_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu3 | fail | 244 | 0.0 | 112.0 | 1736.4 |
| 20260824_calib-idle_node2-gpu3_s0_r02 | calib_idle | completed | nvml.gpu3 | pass | 244 | 0.0 | 64.1 | 2112.4 |
| 20260824_calib-idle_node2-gpu4_s0_r01 | calib_idle | completed | nvml.gpu4 | pass | 292 | 0.0 | 96.6 | 2234.4 |
| 20260824_calib-memcpy_node2-gpu0_s0_r01 | calib_memcpy | completed | nvml.gpu0 | pass | 238 | 0.0 | 81.7 | 2063.4 |
| 20260824_calib-memcpy_node2-gpu1_s0_r01 | calib_memcpy | flagged_channel_health | nvml.gpu1 | fail | 236 | 0.0 | 113.5 | 2143.2 |
| 20260824_calib-memcpy_node2-gpu1_s0_r02 | calib_memcpy | completed | nvml.gpu1 | pass | 237 | 0.0 | 69.5 | 2034.5 |
| 20260824_calib-memcpy_node2-gpu2_s0_r01 | calib_memcpy | flagged_channel_health | nvml.gpu2 | fail | 240 | 0.0 | 296.4 | 1777.5 |
| 20260824_calib-memcpy_node2-gpu2_s0_r02 | calib_memcpy | flagged_channel_health | nvml.gpu2 | fail | 239 | 0.0 | 161.1 | 1891.7 |
| 20260824_calib-memcpy_node2-gpu3_s0_r01 | calib_memcpy | flagged_channel_health | nvml.gpu3 | fail | 251 | 0.0 | 111.8 | 2197.7 |
| 20260824_calib-memcpy_node2-gpu3_s0_r02 | calib_memcpy | completed | nvml.gpu3 | pass | 250 | 0.0 | 67.6 | 2095.1 |
| 20260824_calib-memcpy_node2-gpu4_s0_r01 | calib_memcpy | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 107.6 | None |
| 20260824_calib-memcpy_node2-gpu4_s0_r02 | calib_memcpy | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 56.9 | None |

Channels: 34, failing: 18.
