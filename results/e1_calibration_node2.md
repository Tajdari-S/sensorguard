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
| 20260824_calib-bursty_verifier-gpu0_s0_r01 | calib_bursty | completed | nvml.gpu0 | pass | 256 | 0.0 | 71.8 | 2089.4 |
| 20260824_calib-bursty_verifier-gpu0_s0_r01 | calib_bursty | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-bursty_verifier-gpu1_s0_r01 | calib_bursty | flagged_channel_health | nvml.gpu1 | fail | 252 | 0.0 | 103.9 | 2051.7 |
| 20260824_calib-bursty_verifier-gpu1_s0_r01 | calib_bursty | flagged_channel_health | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-bursty_verifier-gpu1_s0_r02 | calib_bursty | completed | nvml.gpu1 | pass | 252 | 0.0 | 56.4 | 2018.2 |
| 20260824_calib-bursty_verifier-gpu1_s0_r02 | calib_bursty | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-gemm_node2-gpu0_s0_r01 | calib_gemm | completed | nvml.gpu0 | pass | 238 | 0.0 | 52.0 | 2177.4 |
| 20260824_calib-gemm_node2-gpu1_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu1 | fail | 236 | 0.0 | 134.5 | 2185.6 |
| 20260824_calib-gemm_node2-gpu1_s0_r02 | calib_gemm | completed | nvml.gpu1 | pass | 236 | 0.0 | 57.0 | 1962.5 |
| 20260824_calib-gemm_node2-gpu2_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu2 | fail | 240 | 0.0 | 168.0 | 1634.3 |
| 20260824_calib-gemm_node2-gpu2_s0_r02 | calib_gemm | completed | nvml.gpu2 | pass | 240 | 0.0 | 92.1 | 1670.9 |
| 20260824_calib-gemm_node2-gpu3_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu3 | fail | 248 | 0.0 | 110.1 | 2145.9 |
| 20260824_calib-gemm_node2-gpu3_s0_r02 | calib_gemm | completed | nvml.gpu3 | pass | 248 | 0.0 | 62.2 | 2144.2 |
| 20260824_calib-gemm_node2-gpu4_s0_r01 | calib_gemm | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 308 | 0.0 | 102.7 | None |
| 20260824_calib-gemm_node2-gpu4_s0_r02 | calib_gemm | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 296 | 0.0 | 38.5 | None |
| 20260824_calib-gemm_node2-gpu4_s0_r03 | calib_gemm | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 297 | 0.0 | 53.0 | None |
| 20260824_calib-gemm_verifier-gpu0_s0_r01 | calib_gemm | completed | nvml.gpu0 | pass | 238 | 0.0 | 47.7 | 1533.3 |
| 20260824_calib-gemm_verifier-gpu0_s0_r01 | calib_gemm | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-gemm_verifier-gpu0_s0_r02 | calib_gemm | completed | nvml.gpu0 | pass | 238 | 0.0 | 50.7 | 2296.8 |
| 20260824_calib-gemm_verifier-gpu0_s0_r02 | calib_gemm | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-gemm_verifier-gpu0_s0_r03 | calib_gemm | completed | nvml.gpu0 | pass | 238 | 0.0 | 17.7 | 2374.8 |
| 20260824_calib-gemm_verifier-gpu0_s0_r03 | calib_gemm | completed | dcgm.all | pass | 476 | None | None | None |
| 20260824_calib-gemm_verifier-gpu1_s0_r01 | calib_gemm | flagged_channel_health | nvml.gpu1 | fail | 235 | 0.0 | 106.4 | 1712.4 |
| 20260824_calib-gemm_verifier-gpu1_s0_r01 | calib_gemm | flagged_channel_health | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-gemm_verifier-gpu1_s0_r02 | calib_gemm | completed | nvml.gpu1 | pass | 235 | 0.0 | 71.4 | 1515.1 |
| 20260824_calib-gemm_verifier-gpu1_s0_r02 | calib_gemm | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-idle_node2-gpu0_s0_r01 | calib_idle | completed | nvml.gpu0 | pass | 235 | 0.0 | 75.7 | 2250.2 |
| 20260824_calib-idle_node2-gpu1_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu1 | fail | 234 | 0.0 | 135.3 | 2050.2 |
| 20260824_calib-idle_node2-gpu1_s0_r02 | calib_idle | completed | nvml.gpu1 | pass | 234 | 0.0 | 31.2 | 2199.3 |
| 20260824_calib-idle_node2-gpu2_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu2 | fail | 238 | 0.0 | 167.4 | 2125.9 |
| 20260824_calib-idle_node2-gpu2_s0_r02 | calib_idle | completed | nvml.gpu2 | pass | 237 | 0.0 | 47.1 | 2176.0 |
| 20260824_calib-idle_node2-gpu3_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu3 | fail | 244 | 0.0 | 112.0 | 1736.4 |
| 20260824_calib-idle_node2-gpu3_s0_r02 | calib_idle | completed | nvml.gpu3 | pass | 244 | 0.0 | 64.1 | 2112.4 |
| 20260824_calib-idle_node2-gpu4_s0_r01 | calib_idle | completed | nvml.gpu4 | pass | 292 | 0.0 | 96.6 | 2234.4 |
| 20260824_calib-idle_verifier-gpu0_s0_r01 | calib_idle | completed | nvml.gpu0 | pass | 237 | 0.0 | 51.0 | 2180.5 |
| 20260824_calib-idle_verifier-gpu0_s0_r01 | calib_idle | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-idle_verifier-gpu1_s0_r01 | calib_idle | flagged_channel_health | nvml.gpu1 | fail | 235 | 0.0 | 161.5 | 1396.3 |
| 20260824_calib-idle_verifier-gpu1_s0_r01 | calib_idle | flagged_channel_health | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-idle_verifier-gpu1_s0_r02 | calib_idle | completed | nvml.gpu1 | pass | 235 | 0.0 | 53.7 | 2007.4 |
| 20260824_calib-idle_verifier-gpu1_s0_r02 | calib_idle | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-memcpy_node2-gpu0_s0_r01 | calib_memcpy | completed | nvml.gpu0 | pass | 238 | 0.0 | 81.7 | 2063.4 |
| 20260824_calib-memcpy_node2-gpu1_s0_r01 | calib_memcpy | flagged_channel_health | nvml.gpu1 | fail | 236 | 0.0 | 113.5 | 2143.2 |
| 20260824_calib-memcpy_node2-gpu1_s0_r02 | calib_memcpy | completed | nvml.gpu1 | pass | 237 | 0.0 | 69.5 | 2034.5 |
| 20260824_calib-memcpy_node2-gpu2_s0_r01 | calib_memcpy | flagged_channel_health | nvml.gpu2 | fail | 240 | 0.0 | 296.4 | 1777.5 |
| 20260824_calib-memcpy_node2-gpu2_s0_r02 | calib_memcpy | flagged_channel_health | nvml.gpu2 | fail | 239 | 0.0 | 161.1 | 1891.7 |
| 20260824_calib-memcpy_node2-gpu2_s0_r03 | calib_memcpy | completed | nvml.gpu2 | pass | 239 | 0.0 | 74.6 | 2131.9 |
| 20260824_calib-memcpy_node2-gpu3_s0_r01 | calib_memcpy | flagged_channel_health | nvml.gpu3 | fail | 251 | 0.0 | 111.8 | 2197.7 |
| 20260824_calib-memcpy_node2-gpu3_s0_r02 | calib_memcpy | completed | nvml.gpu3 | pass | 250 | 0.0 | 67.6 | 2095.1 |
| 20260824_calib-memcpy_node2-gpu4_s0_r01 | calib_memcpy | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 107.6 | None |
| 20260824_calib-memcpy_node2-gpu4_s0_r02 | calib_memcpy | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 56.9 | None |
| 20260824_calib-memcpy_node2-gpu4_s0_r03 | calib_memcpy | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 70.3 | None |
| 20260824_calib-memcpy_verifier-gpu0_s0_r01 | calib_memcpy | completed | nvml.gpu0 | pass | 238 | 0.0 | 47.6 | 1874.5 |
| 20260824_calib-memcpy_verifier-gpu0_s0_r01 | calib_memcpy | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_calib-memcpy_verifier-gpu1_s0_r01 | calib_memcpy | completed | nvml.gpu1 | pass | 235 | 0.0 | 96.2 | 2274.0 |
| 20260824_calib-memcpy_verifier-gpu1_s0_r01 | calib_memcpy | completed | dcgm.all | pass | 4 | None | None | None |
| 20260824_hpc_fft_verifier-gpu0_s0_r01 | hpc_fft | completed | nvml.gpu0 | pass | 479 | 0.0 | 8.5 | 1962.5 |
| 20260824_hpc_fft_verifier-gpu0_s0_r01 | hpc_fft | completed | dcgm.all | pass | 956 | None | None | None |
| 20260824_hpc_fft_verifier-gpu1_s1_r02 | hpc_fft | completed | nvml.gpu1 | pass | 476 | 0.0 | 13.7 | 2007.0 |
| 20260824_hpc_fft_verifier-gpu1_s1_r02 | hpc_fft | completed | dcgm.all | pass | 950 | None | None | None |
| 20260824_hpc_gemm_verifier-gpu0_s0_r01 | hpc_gemm | completed | nvml.gpu0 | pass | 481 | 0.0 | 8.3 | 1883.4 |
| 20260824_hpc_gemm_verifier-gpu0_s0_r01 | hpc_gemm | completed | dcgm.all | pass | 960 | None | None | None |
| 20260824_hpc_gemm_verifier-gpu1_s1_r02 | hpc_gemm | completed | nvml.gpu1 | pass | 477 | 0.0 | 41.9 | 1938.4 |
| 20260824_hpc_gemm_verifier-gpu1_s1_r02 | hpc_gemm | completed | dcgm.all | pass | 952 | None | None | None |
| 20260824_hpc_memcpy_verifier-gpu0_s0_r01 | hpc_memcpy | completed | nvml.gpu0 | pass | 479 | 0.0 | 8.7 | 1914.4 |
| 20260824_hpc_memcpy_verifier-gpu0_s0_r01 | hpc_memcpy | completed | dcgm.all | pass | 956 | None | None | None |
| 20260824_hpc_memcpy_verifier-gpu1_s1_r02 | hpc_memcpy | completed | nvml.gpu1 | pass | 476 | 0.0 | 13.6 | 1840.7 |
| 20260824_hpc_memcpy_verifier-gpu1_s1_r02 | hpc_memcpy | completed | dcgm.all | pass | 950 | None | None | None |
| 20260824_idle_verifier-gpu0_s0_r01 | idle | completed | nvml.gpu0 | pass | 478 | 0.0 | 27.8 | 2006.3 |
| 20260824_idle_verifier-gpu0_s0_r01 | idle | completed | dcgm.all | pass | 954 | None | None | None |
| 20260824_idle_verifier-gpu1_s1_r02 | idle | completed | nvml.gpu1 | pass | 476 | 0.0 | 13.7 | 2035.8 |
| 20260824_idle_verifier-gpu1_s1_r02 | idle | completed | dcgm.all | pass | 950 | None | None | None |
| 20260824_infer_gpt2_verifier-gpu0_s0_r01 | infer_gpt2 | completed | nvml.gpu0 | pass | 787 | 0.0 | 14.3 | 1856.2 |
| 20260824_infer_gpt2_verifier-gpu0_s0_r01 | infer_gpt2 | completed | dcgm.all | pass | 1572 | None | None | None |
| 20260824_infer_gpt2_verifier-gpu1_s1_r02 | infer_gpt2 | completed | nvml.gpu1 | pass | 783 | 0.0 | 8.6 | 2201.8 |
| 20260824_infer_gpt2_verifier-gpu1_s1_r02 | infer_gpt2 | completed | dcgm.all | pass | 1564 | None | None | None |
| 20260824_infer_resnet_verifier-gpu0_s0_r01 | infer_resnet | completed | nvml.gpu0 | pass | 782 | 0.0 | 20.4 | 2021.0 |
| 20260824_infer_resnet_verifier-gpu0_s0_r01 | infer_resnet | completed | dcgm.all | pass | 1562 | None | None | None |
| 20260824_infer_resnet_verifier-gpu1_s1_r02 | infer_resnet | completed | nvml.gpu1 | pass | 779 | 0.0 | 8.5 | 2270.8 |
| 20260824_infer_resnet_verifier-gpu1_s1_r02 | infer_resnet | completed | dcgm.all | pass | 1554 | None | None | None |
| 20260824_train_gpt2_wikitext_verifier-gpu0_s0_r01 | train_gpt2_wikitext | completed | nvml.gpu0 | pass | 787 | 0.0 | 10.3 | 1970.3 |
| 20260824_train_gpt2_wikitext_verifier-gpu0_s0_r01 | train_gpt2_wikitext | completed | dcgm.all | pass | 1570 | None | None | None |
| 20260824_train_gpt2_wikitext_verifier-gpu1_s1_r02 | train_gpt2_wikitext | completed | nvml.gpu1 | pass | 784 | 0.0 | 10.3 | 1404.2 |
| 20260824_train_gpt2_wikitext_verifier-gpu1_s1_r02 | train_gpt2_wikitext | completed | dcgm.all | pass | 1564 | None | None | None |
| 20260824_train_resnet_cifar10_verifier-gpu0_s0_r01 | train_resnet_cifar10 | completed | nvml.gpu0 | pass | 781 | 0.0 | 11.6 | 2071.5 |
| 20260824_train_resnet_cifar10_verifier-gpu0_s0_r01 | train_resnet_cifar10 | completed | dcgm.all | pass | 1560 | None | None | None |
| 20260824_train_resnet_cifar10_verifier-gpu1_s1_r02 | train_resnet_cifar10 | completed | nvml.gpu1 | pass | 779 | 0.0 | 10.3 | 1952.8 |
| 20260824_train_resnet_cifar10_verifier-gpu1_s1_r02 | train_resnet_cifar10 | completed | dcgm.all | pass | 1554 | None | None | None |
| 20260825_calib-bursty_node2-gpu4_s0_r04 | calib_bursty | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 64.2 | None |
| 20260825_calib-bursty_node2-gpu4_s0_r05 | calib_bursty | completed | nvml.gpu4 | pass | 97 | 0.0 | 69.6 | 1216.2 |
| 20260825_calib-gemm_node2-gpu4_s0_r04 | calib_gemm | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 300 | 0.0 | 48.4 | None |
| 20260825_calib-gemm_node2-gpu4_s0_r05 | calib_gemm | completed | nvml.gpu4 | pass | 97 | 0.0 | 67.7 | 1249.9 |
| 20260825_calib-memcpy_node2-gpu2_s0_r04 | calib_memcpy | completed | nvml.gpu2 | pass | 238 | 0.0 | 61.0 | 1992.7 |
| 20260825_calib-memcpy_node2-gpu4_s0_r04 | calib_memcpy | failed:Command '['/home/felkru/sg-venv/bin/python', '/home/felkru/sensorguard/scripts/loggers/load_marker.py', '--device', 'cuda:4']' timed out after 120 seconds | nvml.gpu4 | fail | 126 | 0.0 | 62.4 | None |
| 20260825_calib-memcpy_node2-gpu4_s0_r05 | calib_memcpy | completed | nvml.gpu4 | pass | 97 | 0.0 | 41.8 | 2013.0 |
| 20260825_hpc_fft_node2-gpu0_s0_r01 | hpc_fft | completed | nvml.gpu0 | pass | 479 | 0.0 | 31.4 | 2099.0 |
| 20260825_hpc_fft_node2-gpu1_s1_r02 | hpc_fft | completed | nvml.gpu1 | pass | 477 | 0.0 | 65.6 | 2235.8 |
| 20260825_hpc_fft_node2-gpu2_s2_r03 | hpc_fft | completed | nvml.gpu2 | pass | 480 | 0.0 | 37.4 | 1999.5 |
| 20260825_hpc_gemm_node2-gpu0_s0_r01 | hpc_gemm | completed | nvml.gpu0 | pass | 479 | 0.0 | 60.8 | 2191.9 |
| 20260825_hpc_gemm_node2-gpu1_s1_r02 | hpc_gemm | completed | nvml.gpu1 | pass | 478 | 0.0 | 73.0 | 1963.6 |
| 20260825_hpc_gemm_node2-gpu2_s2_r03 | hpc_gemm | completed | nvml.gpu2 | pass | 481 | 0.0 | 46.0 | 2109.3 |
| 20260825_hpc_memcpy_node2-gpu0_s0_r01 | hpc_memcpy | completed | nvml.gpu0 | pass | 478 | 0.0 | 45.5 | 2038.7 |
| 20260825_hpc_memcpy_node2-gpu1_s1_r02 | hpc_memcpy | completed | nvml.gpu1 | pass | 477 | 0.0 | 43.9 | 2027.6 |
| 20260825_hpc_memcpy_node2-gpu2_s2_r03 | hpc_memcpy | completed | nvml.gpu2 | pass | 480 | 0.0 | 71.5 | 2093.6 |
| 20260825_idle_node2-gpu0_s0_r01 | idle | completed | nvml.gpu0 | pass | 477 | 0.0 | 71.5 | 2138.6 |
| 20260825_idle_node2-gpu1_s1_r02 | idle | completed | nvml.gpu1 | pass | 475 | 0.0 | 66.9 | 2121.9 |
| 20260825_idle_node2-gpu2_s2_r03 | idle | completed | nvml.gpu2 | pass | 479 | 0.0 | 50.9 | 1930.4 |
| 20260825_infer_gpt2_node2-gpu0_s0_r01 | infer_gpt2 | completed | nvml.gpu0 | pass | 787 | 0.0 | 31.5 | 2005.3 |
| 20260825_infer_gpt2_node2-gpu1_s1_r02 | infer_gpt2 | completed | nvml.gpu1 | pass | 785 | 0.0 | 59.9 | 2159.1 |
| 20260825_infer_gpt2_node2-gpu2_s2_r03 | infer_gpt2 | completed | nvml.gpu2 | pass | 788 | 0.0 | 72.5 | 1646.6 |
| 20260825_infer_resnet_node2-gpu0_s0_r01 | infer_resnet | completed | nvml.gpu0 | pass | 782 | 0.0 | 66.1 | 2177.3 |
| 20260825_infer_resnet_node2-gpu1_s1_r02 | infer_resnet | completed | nvml.gpu1 | pass | 780 | 0.0 | 43.2 | 1958.4 |
| 20260825_infer_resnet_node2-gpu2_s2_r03 | infer_resnet | completed | nvml.gpu2 | pass | 783 | 0.0 | 68.8 | 2174.8 |
| 20260825_train_gpt2_wikitext_node2-gpu0_s0_r01 | train_gpt2_wikitext | completed | nvml.gpu0 | pass | 786 | 0.0 | 45.5 | 2152.3 |
| 20260825_train_gpt2_wikitext_node2-gpu1_s1_r02 | train_gpt2_wikitext | completed | nvml.gpu1 | pass | 784 | 0.0 | 71.6 | 2116.2 |
| 20260825_train_gpt2_wikitext_node2-gpu2_s2_r03 | train_gpt2_wikitext | completed | nvml.gpu2 | pass | 788 | 0.0 | 40.5 | 2235.9 |
| 20260825_train_resnet_cifar10_node2-gpu0_s0_r01 | train_resnet_cifar10 | completed | nvml.gpu0 | pass | 780 | 0.0 | 50.5 | 1895.4 |
| 20260825_train_resnet_cifar10_node2-gpu1_s1_r02 | train_resnet_cifar10 | completed | nvml.gpu1 | pass | 778 | 0.0 | 34.3 | 2085.9 |
| 20260825_train_resnet_cifar10_node2-gpu2_s2_r03 | train_resnet_cifar10 | completed | nvml.gpu2 | pass | 782 | 0.0 | 63.2 | 2200.3 |

Channels: 126, failing: 26.
