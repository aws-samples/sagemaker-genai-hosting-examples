# SageMaker Inference Benchmark Report
**Generated**: 2026-04-06 10:33:35
**Data points**: 1086

## Peak Throughput (Highest Concurrency)

| Model | Instance | Optimization | Use Case | tok/s | RPS | Agg tok/s | p50 (ms) | $/M tokens |
|-------|----------|-------------|----------|-------|-----|-----------|----------|-----------|
| qwen3.5-122b-vanilla | p5e.48xl |  | multiturn_chat | 132.9 | 7.09 | 4254.3 | 4515.5 |  |
| qwen3 | p5e.48xl | 1845 | tool_calling | 126.7 | 7.53 | 4058.7 | 4399.6 |  |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | tool_calling | 123.3 | 7.79 | 3977.6 | 4321.7 |  |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | multiturn_chat | 116.5 | 6.2 | 3720.0 | 5200.5 |  |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | long_context | 110.3 | 5.85 | 3510.0 | 5439.3 |  |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 97.0 | 5.17 | 3102.0 | 6193.4 |  |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 96.6 | 5.15 | 3090.0 | 6210.9 |  |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 93.4 | 6.28 | 3011.3 | 5545.1 |  |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 93.5 | 4.99 | 2994.0 | 6415.7 |  |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 93.1 | 4.96 | 2976.0 | 6436.1 |  |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 92.5 | 5.59 | 2962.1 | 6326.2 |  |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 88.8 | 4.74 | 2844.0 | 6762.4 |  |
| qwen3.5-122b-vanilla | p5e.48xl |  | multiturn_chat | 176.1 | 4.7 | 2817.5 | 3407.2 |  |
| qwen3-235b-eagle3 | p5e.48xl |  | tool_calling | 80.5 | 8.92 | 2779.6 | 3839.2 |  |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 82.7 | 5.89 | 2659.3 | 5095.7 |  |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 81.9 | 4.37 | 2622.0 | 7323.8 |  |
| qwen3-235b-eagle3 | p5e.48xl |  | long_context | 73.5 | 3.91 | 2344.0 | 8156.3 |  |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 73.0 | 3.89 | 2334.0 | 8219.2 | $1.56 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 70.9 | 4.45 | 2282.4 | 7650.5 | $1.60 |
| qwen3.5-122b-vanilla | p5e.48xl |  | long_context | 68.3 | 3.64 | 2184.3 | 8790.8 |  |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 68.4 | 3.64 | 2184.0 | 8759.0 | $1.67 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 67.3 | 3.59 | 2154.0 | 8989.3 | $1.69 |
| qwen3-235b-eagle3 | p5e.48xl |  | multiturn_chat | 66.2 | 3.53 | 2115.1 | 9146.7 |  |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 63.2 | 3.35 | 2010.0 | 9469.3 | $1.81 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 61.0 | 3.22 | 1932.0 | 10107.0 | $1.89 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 55.5 | 2.96 | 1776.0 | 10819.2 | $2.05 |
| kimi-k2.5 | p5e.48xl |  | tool_calling | 53.0 | 5.5 | 1762.8 | 4415.3 |  |
| kimi-k2.5-prefcache | p5e.48xl |  | tool_calling | 53.0 | 5.95 | 1757.6 | 5412.1 |  |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 54.4 | 2.9 | 1740.0 | 11042.8 | $2.09 |
| qwen3-235b | p5e.48xl | vanilla | tool_calling | 51.7 | 5.2 | 1733.7 | 5970.9 |  |
| qwen3-235b-vanilla | p5e.48xl |  | multiturn_chat | 54.0 | 2.88 | 1727.2 | 11170.5 |  |
| qwen3-235b-vanilla | p5e.48xl |  | long_context | 53.9 | 2.87 | 1724.4 | 11173.5 |  |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 53.5 | 2.85 | 1710.0 | 11228.2 | $2.13 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 53.1 | 2.83 | 1698.0 | 11326.8 | $2.15 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 52.8 | 2.82 | 1692.0 | 11357.9 | $2.15 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 52.7 | 2.81 | 1686.0 | 11397.7 | $2.16 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 52.2 | 3.16 | 1676.7 | 10911.5 | $2.17 |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | multiturn_chat | 51.9 | 2.77 | 1662.0 | 11558.3 | $2.19 |
| kimi-k2.5-prefcache | p5e.48xl |  | long_context | 49.3 | 2.63 | 1578.8 | 12221.6 |  |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | tool_calling | 49.1 | 3.36 | 1578.5 | 10217.5 | $2.31 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 49.4 | 2.63 | 1578.0 | 12170.1 | $2.31 |
| qwen3-235b-vanilla | p5e.48xl |  | tool_calling | 44.8 | 4.78 | 1570.5 | 7197.9 |  |
| kimi-k2.5 | p5e.48xl |  | multiturn_chat | 48.9 | 2.61 | 1565.4 | 12258.8 |  |
| kimi-k2.5 | p5e.48xl |  | long_context | 48.7 | 2.6 | 1558.4 | 12342.6 |  |
| kimi-k2.5-prefcache | p5e.48xl |  | multiturn_chat | 48.1 | 2.57 | 1539.9 | 12473.7 |  |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 49.1 | 3.01 | 1534.8 | 11215.5 | $2.37 |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | long_context | 47.3 | 2.52 | 1512.0 | 12678.3 | $2.41 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 40.5 | 2.14 | 1294.7 | 14987.1 |  |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 38.4 | 5.61 | 1191.6 | 4160.2 |  |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | multiturn_chat | 31.6 | 1.69 | 1014.0 | 18977.9 |  |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 31.7 | 1.69 | 1014.0 | 18925.7 |  |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 31.3 | 2.08 | 1003.4 | 17566.7 |  |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | multiturn_chat | 31.3 | 1.67 | 1002.0 | 19169.2 |  |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | tool_calling | 30.9 | 1.88 | 989.3 | 18407.4 |  |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 30.7 | 1.64 | 984.0 | 19562.9 |  |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | long_context | 30.4 | 1.62 | 972.0 | 19734.2 |  |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | long_context | 30.3 | 1.62 | 972.0 | 19790.9 |  |
| gpt-oss-20b | g5.12xl | g5 | long_context | 29.6 | 1.58 | 948.0 | 20270.2 |  |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | tool_calling | 27.2 | 1.77 | 872.8 | 19219.2 |  |
| gpt-oss-20b | g5.12xl | g5 | long_context | 26.6 | 1.42 | 852.0 | 22547.9 |  |
| qwen3-32b-g7e | g7e.2xl | eagle3 | tool_calling | 26.4 | 1.44 | 844.1 | 22750.2 |  |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 24.7 | 1.52 | 793.0 | 23702.8 |  |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | tool_calling | 21.9 | 2.73 | 749.4 | 8344.0 | $4.86 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 46.3 | 1.29 | 740.5 | 12404.1 |  |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | tool_calling | 20.7 | 3.29 | 717.2 | 7911.5 | $5.08 |
| qwen3-32b-g7e | g7e.2xl | eagle3 | multiturn_chat | 22.0 | 1.17 | 702.0 | 27582.5 |  |
| qwen3-32b-g7e | g7e.2xl | eagle3 | long_context | 21.3 | 1.14 | 684.0 | 28351.0 |  |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | tool_calling | 18.7 | 3.2 | 681.6 | 9702.9 |  |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | tool_calling | 20.9 | 1.13 | 672.1 | 28641.9 | $5.42 |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | multiturn_chat | 21.0 | 1.12 | 672.0 | 28647.5 | $5.42 |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | multiturn_chat | 20.9 | 1.11 | 666.0 | 28722.2 | $5.47 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | multiturn_chat | 20.7 | 1.1 | 660.0 | 28969.7 | $5.52 |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | long_context | 20.6 | 1.1 | 660.0 | 29167.6 | $5.52 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | long_context | 19.6 | 1.05 | 630.0 | 30606.1 | $5.78 |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | tool_calling | 18.7 | 2.09 | 615.3 | 15007.5 |  |
| qwen3 | g7e.2xl | lmcache | tool_calling | 18.4 | 2.45 | 613.7 | 11157.8 |  |
| qwen3 | g7e.2xl | lmcache | multiturn_chat | 19.0 | 1.01 | 606.0 | 31612.7 |  |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | multiturn_chat | 19.0 | 1.01 | 606.0 | 31632.7 |  |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | long_context | 18.1 | 0.97 | 582.0 | 33158.9 |  |
| qwen3 | g7e.2xl | g7e | long_context | 18.2 | 0.97 | 582.0 | 33004.4 |  |
| qwen3 | g7e.2xl | lmcache | long_context | 18.0 | 0.96 | 576.0 | 33251.4 |  |
| qwen3 | g7e.2xl | vanilla | tool_calling | 16.9 | 2.59 | 568.8 | 9185.0 |  |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | multiturn_chat | 16.5 | 0.88 | 528.0 | 36417.9 | $6.90 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 16.1 | 0.86 | 516.0 | 37429.9 |  |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | tool_calling | 14.3 | 1.88 | 502.1 | 13550.7 |  |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | tool_calling | 15.8 | 0.84 | 501.5 | 38314.7 | $7.27 |
| overnight-qwen3-lmcache | g5.12xl | g512xl | tool_calling | 14.4 | 2.11 | 496.9 | 11338.4 |  |
| qwen3-prefix-cache | g5.12xl | g512xl | tool_calling | 14.3 | 1.86 | 486.4 | 13198.6 |  |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | tool_calling | 13.9 | 2.01 | 481.4 | 13358.8 |  |
| overnight-qwen3-vanilla | g5.12xl | g512xl | tool_calling | 14.3 | 1.68 | 480.0 | 17855.9 |  |
| qwen3-prefix-cache | g5.12xl | g512xl | multiturn_chat | 13.6 | 0.72 | 432.0 | 44159.8 |  |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | multiturn_chat | 13.6 | 0.72 | 432.0 | 44271.2 |  |
| overnight-qwen3-lmcache | g5.12xl | g512xl | multiturn_chat | 13.6 | 0.72 | 432.0 | 44134.0 |  |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | multiturn_chat | 13.6 | 0.72 | 432.0 | 44182.4 |  |
| overnight-qwen3-vanilla | g5.12xl | g512xl | multiturn_chat | 13.2 | 0.7 | 420.0 | 44348.7 |  |
| overnight-qwen3-lmcache | g5.12xl | g512xl | long_context | 13.1 | 0.7 | 420.0 | 45942.8 |  |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | multiturn_chat | 13.1 | 0.7 | 420.0 | 46394.1 |  |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | long_context | 13.0 | 0.7 | 420.0 | 46086.4 |  |
| overnight-qwen3-vanilla | g5.12xl | g512xl | long_context | 13.0 | 0.69 | 414.0 | 46097.1 |  |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | long_context | 13.0 | 0.69 | 414.0 | 46112.1 |  |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | long_context | 25.6 | 0.68 | 408.0 | 23525.6 | $8.93 |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | long_context | 12.4 | 0.66 | 396.0 | 48520.7 |  |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | multiturn_chat | 11.5 | 0.61 | 366.0 | 52292.7 |  |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | long_context | 20.7 | 0.55 | 330.0 | 29033.2 |  |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | tool_calling | 32.9 | 0.22 | 130.9 | 18493.7 |  |
| qwen3-prefix-cache | g5.12xl | g512xl | long_context | 22.7 | 0.04 | 24.0 | 26380.3 |  |
| qwen3-235b | p5e.48xl | vanilla | multiturn_chat | 0 | 2.89 | 0.0 | 11142.8 |  |
| qwen3-235b | p5e.48xl | vanilla | long_context | 0 | 2.83 | 0.0 | 11160.4 |  |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | long_context | 0 | 4.04 | 0.0 | 7916.5 |  |

## Single Request Performance (C=1)

| Model | Instance | Optimization | Use Case | tok/s | Latency p50 (ms) | Avg Input | Avg Output |
|-------|----------|-------------|----------|-------|-----------------|-----------|------------|
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | long_context | 92.7 | 6467.0 | 1642.0 | 600 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | long_context | 72.2 | 8230.7 | 1659.5 | 600 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | long_context | 57.0 | 9908.8 | 1722.5 | 600 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | multiturn_chat | 94.0 | 6357.7 | 481.4 | 600 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | multiturn_chat | 58.8 | 10246.6 | 539.9 | 600 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | multiturn_chat | 67.3 | 9231.6 | 508.3 | 600 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | tool_calling | 72.5 | 6288.3 | 664.9 | 452.9 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | tool_calling | 59.4 | 7692.8 | 665 | 466.6 |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | tool_calling | 81.3 | 6205.0 | 666.1 | 471.9 |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | long_context | 61.6 | 9794.4 | 1702.3 | 600 |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | long_context | 92.8 | 6461.4 | 1657.2 | 600 |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | multiturn_chat | 94.3 | 6345.4 | 463.3 | 600 |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | multiturn_chat | 60.4 | 9556.2 | 554.7 | 600 |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | tool_calling | 88.2 | 5663.9 | 664.8 | 478.8 |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | tool_calling | 60.4 | 8490.4 | 664.5 | 451.9 |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | long_context | 45.3 | 12853.5 | 1658.3 | 600 |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | multiturn_chat | 59.1 | 10225.6 | 517.7 | 600 |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | multiturn_chat | 39.2 | 13978.3 | 517.7 | 600 |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | multiturn_chat | 52.1 | 11532.7 | 517.7 | 600 |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | tool_calling | 59.5 | 10191.5 | 614.7 | 600 |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | tool_calling | 49.6 | 12318.1 | 614.7 | 591.5 |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | long_context | 36.9 | 16251.8 | 1658.3 | 600 |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | multiturn_chat | 37.3 | 16081.9 | 517.7 | 600 |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | tool_calling | 37.3 | 16068.7 | 614.7 | 592.7 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | long_context | 20.7 | 28774.6 | 1662.3 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | long_context | 20.2 | 29225.2 | 1662.3 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 21.3 | 28214.0 | 521.7 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 21.3 | 28232.7 | 521.7 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 20.7 | 28575.5 | 521.7 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 21.1 | 28339.6 | 521.7 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 19.6 | 29880.2 | 521.7 | 600 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | tool_calling | 19.7 | 8203.1 | 618.7 | 258.9 |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | tool_calling | 19.9 | 9257.7 | 618.7 | 242.0 |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | long_context | 22.0 | 27290.6 | 1662.3 | 600 |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | multiturn_chat | 22.1 | 27180.7 | 484.3 | 600 |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | multiturn_chat | 22.0 | 27216.5 | 521.7 | 600 |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | tool_calling | 21.9 | 9093.4 | 618.7 | 260.1 |
| gpt-oss-20b | g5.12xl | g5 | long_context | 55.4 | 10132.8 | 1621.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | long_context | 97.6 | 6135.2 | 1621.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | long_context | 52.0 | 11066.9 | 1621.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | long_context | 44.1 | 12791.3 | 1621.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | long_context | 97.6 | 6147.6 | 1740.6 | 600 |
| gpt-oss-20b | g5.12xl | g5 | long_context | 54.1 | 10453.4 | 1621.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 49.5 | 12770.6 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 81.3 | 7053.7 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 55.8 | 10142.6 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 59.4 | 10036.5 | 512.8 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 55.3 | 10210.9 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 98.5 | 6082.5 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 51.0 | 10768.7 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 98.5 | 6067.0 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 41.9 | 14488.5 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 71.9 | 7832.7 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 59.1 | 10180.6 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 47.0 | 12734.4 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 86.4 | 6082.9 | 568.7 | 600 |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 98.6 | 6064.2 | 461.6 | 600 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 53.7 | 10036.5 | 668.7 | 539.1 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 98.0 | 5404.1 | 665.5 | 519.9 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 78.7 | 7627.4 | 668.7 | 529.3 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 93.0 | 6077.0 | 668.7 | 541.8 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 98.2 | 6072.1 | 668.7 | 518.6 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 56.0 | 9898.9 | 668.7 | 532.9 |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 42.3 | 13025.2 | 668.7 | 536.8 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 149.8 | 4004.0 | 1723.9 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 124.9 | 4702.1 | 1617.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 148.9 | 4007.1 | 1617.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 149.7 | 4003.1 | 1617.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 114.0 | 5243.3 | 1670.5 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 124.1 | 5046.0 | 1593.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 151.3 | 3957.4 | 587.4 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 150.7 | 3957.7 | 540.4 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 110.1 | 5247.2 | 566.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 130.7 | 4603.6 | 566.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 100.6 | 6172.5 | 552.6 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 133.5 | 3972.6 | 552.6 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 131.3 | 4646.6 | 611.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 151.2 | 3955.7 | 566.7 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 148.6 | 3871.6 | 664.7 | 499.7 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 148.9 | 3853.7 | 664.7 | 497.8 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 136.9 | 3875.4 | 664.7 | 496.4 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 149.2 | 3404.3 | 678 | 508 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 129.0 | 4652.2 | 678 | 600 |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 121.4 | 3957.8 | 672 | 539.4 |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 238.4 | 2515.9 | 1617.7 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 238.4 | 2514.7 | 1617.7 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 155.5 | 4311.6 | 1624.1 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 236.9 | 2525.0 | 1617.7 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 142.6 | 4377.8 | 1695.9 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 241.8 | 2478.3 | 566.7 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 221.9 | 2693.5 | 508.3 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 242.0 | 2477.4 | 566.7 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 240.8 | 2483.8 | 566.7 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 144.7 | 4237.3 | 538 | 600 |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 239.6 | 2462.6 | 664.7 | 487.5 |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 132.0 | 4306.6 | 666.5 | 523.4 |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 149.0 | 3097.7 | 662 | 485.7 |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 238.8 | 2458.6 | 664.7 | 519.8 |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | long_context | 193.7 | 3062.4 | 1617.7 | 600 |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | multiturn_chat | 211.0 | 2835.1 | 566.7 | 600 |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | tool_calling | 225.6 | 2195.9 | 664.7 | 493.6 |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | long_context | 144.4 | 4130.2 | 1639.7 | 600 |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | multiturn_chat | 145.6 | 4100.9 | 581.5 | 600 |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | tool_calling | 144.9 | 3603.0 | 663.8 | 521.3 |
| kimi-k2.5 | p5e.48xl |  | long_context | 107.9 | 5532.3 | 1536.3 | 600 |
| kimi-k2.5 | p5e.48xl |  | long_context | 108.5 | 5527.5 | 1536.3 | 600 |
| kimi-k2.5 | p5e.48xl |  | multiturn_chat | 109.5 | 5456.9 | 511 | 600 |
| kimi-k2.5 | p5e.48xl |  | multiturn_chat | 83.4 | 6672.1 | 511 | 600 |
| kimi-k2.5 | p5e.48xl |  | tool_calling | 107.3 | 2405.3 | 613.3 | 304.7 |
| kimi-k2.5 | p5e.48xl |  | tool_calling | 106.8 | 2000.9 | 613.3 | 265.9 |
| kimi-k2.5-prefcache | p5e.48xl |  | long_context | 108.3 | 5535.6 | 1536.3 | 600 |
| kimi-k2.5-prefcache | p5e.48xl |  | multiturn_chat | 109.6 | 5464.2 | 511 | 600 |
| kimi-k2.5-prefcache | p5e.48xl |  | tool_calling | 107.1 | 2441.3 | 613.3 | 308.4 |
| qwen3-32b-g7e | g7e.2xl | eagle3 | long_context | 41.9 | 14494.5 | 1658.3 | 600 |
| qwen3-32b-g7e | g7e.2xl | eagle3 | multiturn_chat | 38.4 | 15684.3 | 517.7 | 600 |
| qwen3-32b-g7e | g7e.2xl | eagle3 | tool_calling | 45.1 | 13422.1 | 614.7 | 591.0 |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | long_context | 40.4 | 14911.3 | 1662.3 | 600 |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | multiturn_chat | 40.0 | 14836.0 | 521.7 | 600 |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | tool_calling | 48.2 | 4365.7 | 618.7 | 238.9 |
| overnight-qwen3-lmcache | g5.12xl | g512xl | long_context | 22.7 | 26364.7 | 1662.3 | 600 |
| overnight-qwen3-lmcache | g5.12xl | g512xl | multiturn_chat | 22.8 | 26321.5 | 521.7 | 600 |
| overnight-qwen3-lmcache | g5.12xl | g512xl | tool_calling | 22.5 | 7155.0 | 618.7 | 210.5 |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | long_context | 37.0 | 16186.9 | 1662.3 | 600 |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | multiturn_chat | 37.3 | 16060.4 | 521.7 | 600 |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | multiturn_chat | 37.2 | 16068.1 | 521.7 | 600 |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | tool_calling | 36.5 | 5627.7 | 618.7 | 255.6 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | long_context | 33.4 | 16249.0 | 1662.3 | 600 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | long_context | 37.1 | 16173.3 | 1662.3 | 600 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | multiturn_chat | 31.3 | 18714.9 | 521.7 | 600 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | multiturn_chat | 37.3 | 16084.3 | 521.7 | 600 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | tool_calling | 33.3 | 5834.1 | 618.7 | 236.0 |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | tool_calling | 31.6 | 5786.5 | 618.7 | 251.2 |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | long_context | 22.7 | 26384.7 | 1662.3 | 600 |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | multiturn_chat | 22.8 | 26328.4 | 521.7 | 600 |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | tool_calling | 22.3 | 7346.7 | 618.7 | 260.7 |
| overnight-qwen3-vanilla | g5.12xl | g512xl | long_context | 21.2 | 27927.8 | 1662.3 | 600 |
| overnight-qwen3-vanilla | g5.12xl | g512xl | long_context | 21.2 | 27292.6 | 1662.3 | 600 |
| overnight-qwen3-vanilla | g5.12xl | g512xl | multiturn_chat | 20.9 | 27585.6 | 521.7 | 600 |
| overnight-qwen3-vanilla | g5.12xl | g512xl | multiturn_chat | 22.7 | 26315.7 | 521.7 | 600 |
| overnight-qwen3-vanilla | g5.12xl | g512xl | tool_calling | 21.2 | 7935.6 | 618.7 | 205.9 |
| overnight-qwen3-vanilla | g5.12xl | g512xl | tool_calling | 21.0 | 11376.1 | 618.7 | 272.1 |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | long_context | 22.7 | 26399.3 | 1662.3 | 600 |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | multiturn_chat | 22.7 | 26334.5 | 521.7 | 600 |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | tool_calling | 22.4 | 10005.8 | 618.7 | 272.7 |
| qwen3 | p5e.48xl | 1845 | multiturn_chat | 123.5 | 4637.0 | 526.3 | 600 |
| qwen3 | p5e.48xl | 1845 | tool_calling | 116.2 | 2151.2 | 637.7 | 306.7 |
| qwen3 | g7e.2xl | g7e | long_context | 22.0 | 27274.5 | 1662.3 | 600 |
| qwen3 | g7e.2xl | lmcache | long_context | 22.0 | 27288.9 | 1662.3 | 600 |
| qwen3 | g7e.2xl | lmcache | multiturn_chat | 22.0 | 27217.5 | 521.7 | 600 |
| qwen3 | g7e.2xl | lmcache | tool_calling | 21.8 | 8638.6 | 618.7 | 239.2 |
| qwen3 | g7e.2xl | vanilla | long_context | 19.4 | 30488.8 | 1630.7 | 600 |
| qwen3 | g7e.2xl | vanilla | multiturn_chat | 20.6 | 28948.6 | 521.7 | 600 |
| qwen3 | g7e.2xl | vanilla | multiturn_chat | 19.8 | 30062.2 | 521.7 | 600 |
| qwen3 | g7e.2xl | vanilla | multiturn_chat | 20.5 | 28878.2 | 521.7 | 600 |
| qwen3 | g7e.2xl | vanilla | multiturn_chat | 20.7 | 28667.3 | 521.7 | 600 |
| qwen3 | g7e.2xl | vanilla | multiturn_chat | 20.6 | 29322.4 | 464.4 | 600 |
| qwen3 | g7e.2xl | vanilla | tool_calling | 20.8 | 9574.8 | 618.7 | 246.7 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 48.4 | 12405.3 | 0 | 601.1 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 48.9 | 12321.7 | 0 | 602.5 |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 45.6 | 3084.6 | 0 | 176.0 |
| qwen3-235b | p5e.48xl | vanilla | long_context | 0 | 5427.8 | 0 | 0 |
| qwen3-235b | p5e.48xl | vanilla | long_context | 94.0 | 6263.7 | 1662.3 | 600 |
| qwen3-235b | p5e.48xl | vanilla | long_context | 0 | 6255.8 | 0 | 0 |
| qwen3-235b | p5e.48xl | vanilla | multiturn_chat | 0 | 5382.1 | 0 | 0 |
| qwen3-235b | p5e.48xl | vanilla | multiturn_chat | 0 | 6210.5 | 0 | 0 |
| qwen3-235b | p5e.48xl | vanilla | multiturn_chat | 95.8 | 6213.1 | 521.7 | 600 |
| qwen3-235b | p5e.48xl | vanilla | tool_calling | 89.8 | 3233.9 | 618.7 | 333.7 |
| qwen3-235b | p5e.48xl | vanilla | tool_calling | 0 | 2853.2 | 0 | 0 |
| qwen3-235b | p5e.48xl | vanilla | tool_calling | 0 | 2407.3 | 0 | 0 |
| qwen3-235b-eagle3 | p5e.48xl |  | long_context | 171.6 | 3487.1 | 1662.3 | 600 |
| qwen3-235b-eagle3 | p5e.48xl |  | multiturn_chat | 154.9 | 3878.5 | 521.7 | 600 |
| qwen3-235b-eagle3 | p5e.48xl |  | tool_calling | 198.6 | 1472.9 | 618.7 | 317.5 |
| qwen3-235b-vanilla | p5e.48xl |  | long_context | 111.9 | 5364.3 | 1662.3 | 600 |
| qwen3-235b-vanilla | p5e.48xl |  | long_context | 110.6 | 5364.0 | 1662.3 | 600 |
| qwen3-235b-vanilla | p5e.48xl |  | multiturn_chat | 111.6 | 5327.7 | 521.7 | 600 |
| qwen3-235b-vanilla | p5e.48xl |  | multiturn_chat | 105.2 | 5319.2 | 521.7 | 600 |
| qwen3-235b-vanilla | p5e.48xl |  | tool_calling | 110.0 | 2835.3 | 618.7 | 337.0 |
| qwen3-235b-vanilla | p5e.48xl |  | tool_calling | 105.4 | 2013.3 | 618.7 | 274.1 |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | multiturn_chat | 34.7 | 17378.7 | 460.4 | 600 |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | multiturn_chat | 34.8 | 17312.3 | 517.7 | 600 |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | tool_calling | 39.7 | 15217.5 | 614.7 | 595.3 |
| qwen3-prefix-cache | g5.12xl | g512xl | long_context | 22.7 | 26380.3 | 1559.5 | 600 |
| qwen3-prefix-cache | g5.12xl | g512xl | multiturn_chat | 22.8 | 26300.9 | 517.7 | 600 |
| qwen3-prefix-cache | g5.12xl | g512xl | tool_calling | 22.8 | 26310.9 | 614.7 | 591.0 |
| qwen3-prefix-cache | g5.12xl | g512xl | tool_calling | 22.5 | 7614.9 | 618.7 | 215.0 |
| qwen3.5-122b-vanilla | p5e.48xl |  | long_context | 121.5 | 5165.6 | 1704.3 | 600 |
| qwen3.5-122b-vanilla | p5e.48xl |  | multiturn_chat | 174.4 | 3405.3 | 526.3 | 600 |
| qwen3.5-122b-vanilla | p5e.48xl |  | multiturn_chat | 120.5 | 5229.1 | 526.3 | 600 |
| qwen3.5-122b-vanilla | p5e.48xl |  | multiturn_chat | 133.9 | 4375.2 | 526.3 | 600 |
| qwen3.5-122b-vanilla | p5e.48xl |  | tool_calling | 121.8 | 2185.2 | 637.7 | 301.3 |
| qwen35-122b | p5e.48xl | 1845 | long_context | 124.5 | 4493.8 | 1704.3 | 600 |
| qwen35-122b | p5e.48xl | 1845 | multiturn_chat | 171.6 | 3405.7 | 526.3 | 600 |
| qwen35-122b | p5e.48xl | 1845 | tool_calling | 121.0 | 2200.2 | 637.7 | 321.3 |

## Optimization Speedup (vs Vanilla)

| Model | Instance | Use Case | Vanilla tok/s | Optimization | Optimized tok/s | Speedup |
|-------|----------|----------|--------------|-------------|----------------|---------|
| bench-v2-gptoss20b-vanilla | g5.12xl | long_context | 0.0 | g512xl | 30.3 | N/A |
| bench-v2-gptoss20b-vanilla | g5.12xl | multiturn_chat | 0.0 | g512xl | 31.3 | N/A |
| bench-v2-gptoss20b-vanilla | g5.12xl | tool_calling | 0.0 | g512xl | 27.2 | N/A |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | long_context | 0.0 | 12xl | 30.4 | N/A |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | multiturn_chat | 0.0 | 12xl | 31.6 | N/A |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | tool_calling | 0.0 | 12xl | 30.9 | N/A |
| bench-v2-qwen3-eagle3 | g6e.12xl | long_context | 0.0 | g6e12xl | 0.0 | N/A |
| bench-v2-qwen3-eagle3 | g6e.12xl | multiturn_chat | 0.0 | g6e12xl | 16.5 | N/A |
| bench-v2-qwen3-eagle3 | g6e.12xl | tool_calling | 0.0 | g6e12xl | 15.8 | N/A |
| bench-v2-qwen3-vanilla | g6e.12xl | long_context | 0.0 | g6e12xl | 25.6 | N/A |
| bench-v2-qwen3-vanilla | g7e.2xl | long_context | 0.0 | g7e2xl | 20.7 | N/A |
| bench-v2-qwen3-vanilla | g6e.12xl | multiturn_chat | 0.0 | g6e12xl | 20.9 | N/A |
| bench-v2-qwen3-vanilla | g7e.2xl | multiturn_chat | 0.0 | g7e2xl | 16.1 | N/A |
| bench-v2-qwen3-vanilla | g6e.12xl | tool_calling | 0.0 | g6e12xl | 20.9 | N/A |
| final-qwen3-prefcache | g7e.2xl | long_context | 0.0 | g7e2xl | 18.1 | N/A |
| final-qwen3-prefcache | g7e.2xl | multiturn_chat | 0.0 | g7e2xl | 19.0 | N/A |
| final-qwen3-prefcache | g7e.2xl | tool_calling | 0.0 | g7e2xl | 18.7 | N/A |
| gpt-oss-20b | g5.12xl | long_context | 0.0 | g5 | 26.6 | N/A |
| gpt-oss-20b | g5.12xl | long_context | 0.0 | g5 | 29.6 | N/A |
| gpt-oss-20b | g6e.12xl | long_context | 0.0 | g6e | 52.7 | N/A |
| gpt-oss-20b | g6e.12xl | long_context | 0.0 | g6e | 63.2 | N/A |
| gpt-oss-20b | g6e.12xl | long_context | 0.0 | g6e | 53.1 | N/A |
| gpt-oss-20b | g6e.12xl | long_context | 0.0 | g6e | 53.5 | N/A |
| gpt-oss-20b | g6e.12xl | long_context | 0.0 | g6e | 61.0 | N/A |
| gpt-oss-20b | g7e.2xl | long_context | 0.0 | g7e | 93.1 | N/A |
| gpt-oss-20b | g7e.2xl | long_context | 0.0 | g7e | 93.5 | N/A |
| gpt-oss-20b | g7e.2xl | long_context | 0.0 | g7e | 81.9 | N/A |
| gpt-oss-20b | g5.12xl | multiturn_chat | 0.0 | g5 | 30.7 | N/A |
| gpt-oss-20b | g5.12xl | multiturn_chat | 0.0 | g5 | 31.7 | N/A |
| gpt-oss-20b | g6e.12xl | multiturn_chat | 0.0 | g6e | 54.4 | N/A |
| gpt-oss-20b | g6e.12xl | multiturn_chat | 0.0 | g6e | 49.4 | N/A |
| gpt-oss-20b | g6e.12xl | multiturn_chat | 0.0 | g6e | 67.3 | N/A |
| gpt-oss-20b | g6e.12xl | multiturn_chat | 0.0 | g6e | 68.4 | N/A |
| gpt-oss-20b | g6e.12xl | multiturn_chat | 0.0 | g6e | 55.5 | N/A |
| gpt-oss-20b | g7e.2xl | multiturn_chat | 0.0 | g7e | 96.6 | N/A |
| gpt-oss-20b | g7e.2xl | multiturn_chat | 0.0 | g7e | 88.8 | N/A |
| gpt-oss-20b | g7e.2xl | multiturn_chat | 0.0 | g7e | 97.0 | N/A |
| gpt-oss-20b | g5.12xl | tool_calling | 0.0 | g5 | 24.7 | N/A |
| gpt-oss-20b | g5.12xl | tool_calling | 0.0 | g5 | 31.3 | N/A |
| gpt-oss-20b | g6e.12xl | tool_calling | 0.0 | g6e | 52.2 | N/A |
| gpt-oss-20b | g6e.12xl | tool_calling | 0.0 | g6e | 49.1 | N/A |
| gpt-oss-20b | g6e.12xl | tool_calling | 0.0 | g6e | 70.9 | N/A |
| gpt-oss-20b | g6e.12xl | tool_calling | 0.0 | g6e | 52.8 | N/A |
| gpt-oss-20b | g6e.12xl | tool_calling | 0.0 | g6e | 73.0 | N/A |
| gpt-oss-20b | g7e.2xl | tool_calling | 0.0 | g7e | 92.5 | N/A |
| gpt-oss-20b | g7e.2xl | tool_calling | 0.0 | g7e | 82.7 | N/A |
| gpt-oss-20b | g7e.2xl | tool_calling | 0.0 | g7e | 93.4 | N/A |
| gpt-oss-20b-eagle3 | g7e.2xl | long_context | 0.0 | g7e | 110.3 | N/A |
| gpt-oss-20b-eagle3 | g7e.2xl | multiturn_chat | 0.0 | g7e | 116.5 | N/A |
| gpt-oss-20b-eagle3 | g7e.2xl | tool_calling | 0.0 | g7e | 123.3 | N/A |
| gptoss20b-byoc-vanilla | g6e.12xl | long_context | 0.0 | g6e12xl | 47.3 | N/A |
| gptoss20b-byoc-vanilla | g6e.12xl | multiturn_chat | 0.0 | g6e12xl | 51.9 | N/A |
| gptoss20b-byoc-vanilla | g6e.12xl | tool_calling | 0.0 | g6e12xl | 49.1 | N/A |
| kimi-k2.5 | p5e.48xl | long_context | 0.0 |  | 48.7 | N/A |
| kimi-k2.5 | p5e.48xl | multiturn_chat | 0.0 |  | 48.9 | N/A |
| kimi-k2.5 | p5e.48xl | tool_calling | 0.0 |  | 53.0 | N/A |
| kimi-k2.5-prefcache | p5e.48xl | long_context | 0.0 |  | 49.3 | N/A |
| kimi-k2.5-prefcache | p5e.48xl | multiturn_chat | 0.0 |  | 48.1 | N/A |
| kimi-k2.5-prefcache | p5e.48xl | tool_calling | 0.0 |  | 53.0 | N/A |
| qwen3-32b-g7e | g7e.2xl | long_context | 0.0 | eagle3 | 21.3 | N/A |
| qwen3-32b-g7e | g7e.2xl | multiturn_chat | 0.0 | eagle3 | 22.0 | N/A |
| qwen3-32b-g7e | g7e.2xl | tool_calling | 0.0 | eagle3 | 26.4 | N/A |
| overnight-qwen3-eagle3-g512xl | g5.12xl | long_context | 0.0 | use1 | 12.4 | N/A |
| overnight-qwen3-eagle3-g512xl | g5.12xl | multiturn_chat | 0.0 | use1 | 13.1 | N/A |
| overnight-qwen3-eagle3-g512xl | g5.12xl | tool_calling | 0.0 | use1 | 18.7 | N/A |
| overnight-qwen3-lmcache | g5.12xl | long_context | 0.0 | g512xl | 13.1 | N/A |
| overnight-qwen3-lmcache | g6e.12xl | long_context | 0.0 | g6e12xl | 20.6 | N/A |
| overnight-qwen3-lmcache | g5.12xl | multiturn_chat | 0.0 | g512xl | 13.6 | N/A |
| overnight-qwen3-lmcache | g6e.12xl | multiturn_chat | 0.0 | g6e12xl | 21.0 | N/A |
| overnight-qwen3-lmcache | g5.12xl | tool_calling | 0.0 | g512xl | 14.4 | N/A |
| overnight-qwen3-lmcache | g6e.12xl | tool_calling | 0.0 | g6e12xl | 21.9 | N/A |
| overnight-qwen3-prefcache | g6e.12xl | long_context | 0.0 | g6e12xl | 19.6 | N/A |
| overnight-qwen3-prefcache | g6e.12xl | multiturn_chat | 0.0 | g6e12xl | 20.7 | N/A |
| overnight-qwen3-prefcache | g6e.12xl | tool_calling | 0.0 | g6e12xl | 20.7 | N/A |
| overnight-qwen3-prefcache-g512xl | g5.12xl | long_context | 0.0 | use1 | 13.0 | N/A |
| overnight-qwen3-prefcache-g512xl | g5.12xl | multiturn_chat | 0.0 | use1 | 13.6 | N/A |
| overnight-qwen3-prefcache-g512xl | g5.12xl | tool_calling | 0.0 | use1 | 14.3 | N/A |
| overnight-qwen3-vanilla | g5.12xl | long_context | 0.0 | g512xl | 13.0 | N/A |
| overnight-qwen3-vanilla | g5.12xl | multiturn_chat | 0.0 | g512xl | 13.2 | N/A |
| overnight-qwen3-vanilla | g5.12xl | tool_calling | 0.0 | g512xl | 14.3 | N/A |
| overnight-qwen3-vanilla-g512xl | g5.12xl | long_context | 0.0 | use1 | 13.0 | N/A |
| overnight-qwen3-vanilla-g512xl | g5.12xl | multiturn_chat | 0.0 | use1 | 13.6 | N/A |
| overnight-qwen3-vanilla-g512xl | g5.12xl | tool_calling | 0.0 | use1 | 13.9 | N/A |
| qwen3 | g7e.2xl | long_context | 0.0 | g7e | 18.2 | N/A |
| qwen3 | g7e.2xl | long_context | 0.0 | lmcache | 18.0 | N/A |
| qwen3 | g7e.2xl | multiturn_chat | 0.0 | lmcache | 19.0 | N/A |
| qwen3 | p5e.48xl | tool_calling | 0.0 | 1845 | 126.7 | N/A |
| qwen3 | g7e.2xl | tool_calling | 16.9 | lmcache | 18.4 | 1.09x |
| qwen3-235b-eagle3 | p5e.48xl | long_context | 0.0 |  | 73.5 | N/A |
| qwen3-235b-eagle3 | p5e.48xl | multiturn_chat | 0.0 |  | 66.2 | N/A |
| qwen3-235b-eagle3 | p5e.48xl | tool_calling | 0.0 |  | 80.5 | N/A |
| qwen3-235b-vanilla | p5e.48xl | long_context | 0.0 |  | 53.9 | N/A |
| qwen3-235b-vanilla | p5e.48xl | multiturn_chat | 0.0 |  | 54.0 | N/A |
| qwen3-235b-vanilla | p5e.48xl | tool_calling | 0.0 |  | 44.8 | N/A |
| qwen3-eagle3-g512xl | g5.12xl | multiturn_chat | 0.0 | useast1 | 11.5 | N/A |
| qwen3-eagle3-g512xl | g5.12xl | tool_calling | 0.0 | useast1 | 32.9 | N/A |
| qwen3-prefix-cache | g5.12xl | long_context | 0.0 | g512xl | 22.7 | N/A |
| qwen3-prefix-cache | g5.12xl | multiturn_chat | 0.0 | g512xl | 13.6 | N/A |
| qwen3-prefix-cache | g5.12xl | tool_calling | 0.0 | g512xl | 14.3 | N/A |
| qwen3.5-122b-vanilla | p5e.48xl | long_context | 0.0 |  | 68.3 | N/A |
| qwen3.5-122b-vanilla | p5e.48xl | multiturn_chat | 0.0 |  | 176.1 | N/A |
| qwen3.5-122b-vanilla | p5e.48xl | multiturn_chat | 0.0 |  | 132.9 | N/A |

## Time to First Token (TTFT)

| Model | Instance | Optimization | Use Case | C | TTFT p50 (ms) | TTFT p90 (ms) | TTFT avg (ms) |
|-------|----------|-------------|----------|---|--------------|--------------|--------------|
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 1 | 70.0 | 153.9 | 85.7 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 1 | 73.1 | 81.3 | 78.3 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 1 | 72.1 | 84.2 | 80.8 |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 16 | 114.3 | 166.1 | 120.6 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 16 | 72.2 | 72.2 | 72.2 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 16 | 117.6 | 213.2 | 139.1 |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 2 | 105.9 | 199.2 | 118.3 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 2 | 106.3 | 145.3 | 110.6 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 2 | 104.8 | 181.1 | 121.7 |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 32 | 186.6 | 216.5 | 197.4 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 32 | 270.4 | 280.2 | 220.1 |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 4 | 110.0 | 189.7 | 137.3 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 4 | 111.5 | 209.9 | 136.1 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 4 | 109.2 | 205.1 | 130.8 |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 8 | 110.8 | 208.2 | 121.3 |
| qwen3-14b | g7e.2xl | vanilla | long_context | 8 | 105.0 | 108.0 | 96.3 |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 8 | 116.3 | 209.7 | 150.2 |

## Latency Scaling Under Load

| Model | Instance | Optimization | Use Case | C=1 p50 | Peak C p50 | Degradation |
|-------|----------|-------------|----------|---------|-----------|------------|
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | long_context | 9909ms | 22336ms (C=32) | +125% |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | multiturn_chat | 9232ms | 19032ms (C=32) | +106% |
| bench-v2-gptoss20b-vanilla | g5.12xl | g512xl | tool_calling | 6205ms | 17578ms (C=32) | +183% |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | long_context | 6461ms | 19734ms (C=32) | +205% |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | multiturn_chat | 9556ms | 19468ms (C=32) | +104% |
| bench-v2-gptoss20b-vanilla-g5b | g5.12xl | 12xl | tool_calling | 8490ms | 15441ms (C=32) | +82% |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | long_context | 12854ms | 7916ms (C=32) | +-38% |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | multiturn_chat | 11533ms | 38881ms (C=32) | +237% |
| bench-v2-qwen3-eagle3 | g6e.12xl | g6e12xl | tool_calling | 12318ms | 38315ms (C=32) | +211% |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | long_context | 16252ms | 23526ms (C=16) | +45% |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | multiturn_chat | 16082ms | 28722ms (C=32) | +79% |
| bench-v2-qwen3-vanilla | g6e.12xl | g6e12xl | tool_calling | 16069ms | 28642ms (C=32) | +78% |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | long_context | 29225ms | 29033ms (C=16) | +-1% |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | multiturn_chat | 29880ms | 36760ms (C=32) | +23% |
| bench-v2-qwen3-vanilla | g7e.2xl | g7e2xl | tool_calling | 9258ms | 8918ms (C=32) | +-4% |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | long_context | 27291ms | 33159ms (C=32) | +22% |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | multiturn_chat | 27216ms | 31633ms (C=32) | +16% |
| final-qwen3-prefcache | g7e.2xl | g7e2xl | tool_calling | 9093ms | 15008ms (C=32) | +65% |
| gpt-oss-20b | g5.12xl | g5 | long_context | 10453ms | 22669ms (C=32) | +117% |
| gpt-oss-20b | g5.12xl | g5 | multiturn_chat | 6064ms | 18926ms (C=32) | +212% |
| gpt-oss-20b | g5.12xl | g5 | tool_calling | 13025ms | 20093ms (C=32) | +54% |
| gpt-oss-20b | g6e.12xl | g6e | long_context | 5046ms | 10107ms (C=32) | +100% |
| gpt-oss-20b | g6e.12xl | g6e | multiturn_chat | 3956ms | 10819ms (C=32) | +174% |
| gpt-oss-20b | g6e.12xl | g6e | tool_calling | 3958ms | 11309ms (C=32) | +186% |
| gpt-oss-20b | g7e.2xl | g7e | long_context | 4378ms | 7392ms (C=32) | +69% |
| gpt-oss-20b | g7e.2xl | g7e | multiturn_chat | 4237ms | 6670ms (C=32) | +57% |
| gpt-oss-20b | g7e.2xl | g7e | tool_calling | 2459ms | 5545ms (C=32) | +126% |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | long_context | 3062ms | 5439ms (C=32) | +78% |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | multiturn_chat | 2835ms | 5200ms (C=32) | +83% |
| gpt-oss-20b-eagle3 | g7e.2xl | g7e | tool_calling | 2196ms | 4322ms (C=32) | +97% |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | long_context | 4130ms | 12678ms (C=32) | +207% |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | multiturn_chat | 4101ms | 11558ms (C=32) | +182% |
| gptoss20b-byoc-vanilla | g6e.12xl | g6e12xl | tool_calling | 3603ms | 10218ms (C=32) | +184% |
| kimi-k2.5 | p5e.48xl |  | long_context | 5528ms | 12284ms (C=32) | +122% |
| kimi-k2.5 | p5e.48xl |  | multiturn_chat | 6672ms | 12228ms (C=32) | +83% |
| kimi-k2.5 | p5e.48xl |  | tool_calling | 2001ms | 4479ms (C=32) | +124% |
| kimi-k2.5-prefcache | p5e.48xl |  | long_context | 5536ms | 12222ms (C=32) | +121% |
| kimi-k2.5-prefcache | p5e.48xl |  | multiturn_chat | 5464ms | 12474ms (C=32) | +128% |
| kimi-k2.5-prefcache | p5e.48xl |  | tool_calling | 2441ms | 5412ms (C=32) | +122% |
| qwen3-32b-g7e | g7e.2xl | eagle3 | long_context | 14494ms | 28351ms (C=32) | +96% |
| qwen3-32b-g7e | g7e.2xl | eagle3 | multiturn_chat | 15684ms | 27582ms (C=32) | +76% |
| qwen3-32b-g7e | g7e.2xl | eagle3 | tool_calling | 13422ms | 22750ms (C=32) | +69% |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | long_context | 14911ms | 48521ms (C=32) | +225% |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | multiturn_chat | 14836ms | 46394ms (C=32) | +213% |
| overnight-qwen3-eagle3-g512xl | g5.12xl | use1 | tool_calling | 4366ms | 9703ms (C=32) | +122% |
| overnight-qwen3-lmcache | g5.12xl | g512xl | long_context | 26365ms | 45943ms (C=32) | +74% |
| overnight-qwen3-lmcache | g5.12xl | g512xl | multiturn_chat | 26322ms | 44134ms (C=32) | +68% |
| overnight-qwen3-lmcache | g5.12xl | g512xl | tool_calling | 7155ms | 11338ms (C=32) | +58% |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | long_context | 16187ms | 29168ms (C=32) | +80% |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | multiturn_chat | 16068ms | 28648ms (C=32) | +78% |
| overnight-qwen3-lmcache | g6e.12xl | g6e12xl | tool_calling | 5628ms | 8344ms (C=32) | +48% |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | long_context | 16173ms | 29094ms (C=32) | +80% |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | multiturn_chat | 16084ms | 28727ms (C=32) | +79% |
| overnight-qwen3-prefcache | g6e.12xl | g6e12xl | tool_calling | 5786ms | 9034ms (C=32) | +56% |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | long_context | 26385ms | 46112ms (C=32) | +75% |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | multiturn_chat | 26328ms | 44271ms (C=32) | +68% |
| overnight-qwen3-prefcache-g512xl | g5.12xl | use1 | tool_calling | 7347ms | 13551ms (C=32) | +84% |
| overnight-qwen3-vanilla | g5.12xl | g512xl | long_context | 27293ms | 46097ms (C=32) | +69% |
| overnight-qwen3-vanilla | g5.12xl | g512xl | multiturn_chat | 26316ms | 44288ms (C=32) | +68% |
| overnight-qwen3-vanilla | g5.12xl | g512xl | tool_calling | 11376ms | 12250ms (C=32) | +8% |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | long_context | 26399ms | 46086ms (C=32) | +75% |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | multiturn_chat | 26334ms | 44182ms (C=32) | +68% |
| overnight-qwen3-vanilla-g512xl | g5.12xl | use1 | tool_calling | 10006ms | 13359ms (C=32) | +34% |
| qwen3 | g7e.2xl | g7e | long_context | 27274ms | 33004ms (C=32) | +21% |
| qwen3 | g7e.2xl | lmcache | long_context | 27289ms | 33251ms (C=32) | +22% |
| qwen3 | g7e.2xl | lmcache | multiturn_chat | 27218ms | 31613ms (C=32) | +16% |
| qwen3 | g7e.2xl | lmcache | tool_calling | 8639ms | 11158ms (C=32) | +29% |
| qwen3 | g7e.2xl | vanilla | long_context | 30489ms | 28494ms (C=2) | +-7% |
| qwen3 | g7e.2xl | vanilla | multiturn_chat | 29322ms | 37507ms (C=32) | +28% |
| qwen3 | g7e.2xl | vanilla | tool_calling | 9575ms | 9185ms (C=32) | +-4% |
| qwen3 | p5e.48xl | 1845 | multiturn_chat | 4637ms | 8763ms (C=32) | +89% |
| qwen3 | p5e.48xl | 1845 | tool_calling | 2151ms | 4400ms (C=32) | +105% |
| qwen3-14b | g7e.2xl | vanilla | long_context | 12405ms | 12404ms (C=16) | +-0% |
| qwen3-14b | g7e.2xl | vanilla | multiturn_chat | 12322ms | 14987ms (C=32) | +22% |
| qwen3-14b | g7e.2xl | vanilla | tool_calling | 3085ms | 4160ms (C=32) | +35% |
| qwen3-235b | p5e.48xl | vanilla | long_context | 6256ms | 11832ms (C=32) | +89% |
| qwen3-235b | p5e.48xl | vanilla | multiturn_chat | 6213ms | 11237ms (C=32) | +81% |
| qwen3-235b | p5e.48xl | vanilla | tool_calling | 2407ms | 6411ms (C=32) | +166% |
| qwen3-235b-eagle3 | p5e.48xl |  | long_context | 3487ms | 8156ms (C=32) | +134% |
| qwen3-235b-eagle3 | p5e.48xl |  | multiturn_chat | 3878ms | 9147ms (C=32) | +136% |
| qwen3-235b-eagle3 | p5e.48xl |  | tool_calling | 1473ms | 3839ms (C=32) | +161% |
| qwen3-235b-vanilla | p5e.48xl |  | long_context | 5364ms | 11143ms (C=32) | +108% |
| qwen3-235b-vanilla | p5e.48xl |  | multiturn_chat | 5319ms | 11145ms (C=32) | +110% |
| qwen3-235b-vanilla | p5e.48xl |  | tool_calling | 2013ms | 6406ms (C=32) | +218% |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | multiturn_chat | 17312ms | 52293ms (C=32) | +202% |
| qwen3-eagle3-g512xl | g5.12xl | useast1 | tool_calling | 15218ms | 18494ms (C=4) | +22% |
| qwen3-prefix-cache | g5.12xl | g512xl | multiturn_chat | 26301ms | 44160ms (C=32) | +68% |
| qwen3-prefix-cache | g5.12xl | g512xl | tool_calling | 7615ms | 13199ms (C=32) | +73% |
| qwen3.5-122b-vanilla | p5e.48xl |  | long_context | 5166ms | 8791ms (C=32) | +70% |
| qwen3.5-122b-vanilla | p5e.48xl |  | multiturn_chat | 4375ms | 7772ms (C=32) | +78% |
| qwen3.5-122b-vanilla | p5e.48xl |  | tool_calling | 2185ms | 3665ms (C=32) | +68% |
| qwen35-122b | p5e.48xl | 1845 | long_context | 4494ms | 8799ms (C=32) | +96% |
| qwen35-122b | p5e.48xl | 1845 | multiturn_chat | 3406ms | 7531ms (C=32) | +121% |
| qwen35-122b | p5e.48xl | 1845 | tool_calling | 2200ms | 3137ms (C=32) | +43% |
