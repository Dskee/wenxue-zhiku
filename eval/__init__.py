"""问学知库 · 检索评测（离线可复现，指标部分零 LLM 调用）。

目录约定：
- configs/     三份检索配置（vector_only / hybrid / hybrid_rerank），同一份代码只换参数
- dataset/     冻结的评测集（带 gold chunk id），改动必须写进 dataset/CHANGELOG.md
- metrics.py   召回/排序指标纯函数，只依赖标准库
- chunks.py    稳定的 chunk id（sha1(文件名 + 正文)），跨重建不变
- run_ablation.py  跑批并产出 results/YYYY-MM-DD.md
"""