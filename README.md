# TIC-Bench

这是从 DataConstruct 脱敏迁移出的推理与评估工具。仓库只保留：

- Spatial、Visual、Temporal 三类数据的提问消息构造；
- 模型回答的断点续跑；
- 三位 LLM Judge 独立语义评估与 2/3 多数投票；
- 离线准确率与覆盖率汇总；
- 一个最小 OpenAI Responses 调用模板。

本仓库不包含数据构造、图像生成、供应商路由、远程地址、凭据、用量统计、调用日志、模型权重或源数据。DataConstruct 源目录不会被写入。

## 安装

```bash
python -m pip install -e ".[test]"
```

OpenAI 客户端使用其标准运行环境配置。本仓库不保存或接收任何凭据参数。

## 使用

先仅检查消息，不发起模型请求：

```bash
python -m ticbench ask DATASET_ROOT --model MODEL_NAME --dry-run
```

运行回答生成：

```bash
python -m ticbench ask DATASET_ROOT --model MODEL_NAME
```

评估某个回答文件名对应的模型；`--target` 是 `answer_results_` 后、不含 `.json` 的部分：

```bash
python -m ticbench evaluate DATASET_ROOT --judge-model JUDGE_1 JUDGE_2 JUDGE_3 --target TARGET_NAME
```

离线汇总：

```bash
python -m ticbench report DATASET_ROOT
```

三类数据沿用原有目录约定：

- Spatial：`question_*/dataset_caption.jsonl`；
- Visual：`question_*/qa_pairs*.json` 与 `node_sequence*.json`；
- Temporal：`question_*/qa_pairs*.json`，根目录名含 `time` 或使用 `--kind temporal`。

断点续跑会保护所有已有非空回答，不因其历史状态字段而覆盖。写入只发生在传入的 `DATASET_ROOT` 内。

评估要求三个不同的裁判模型。输出的每条记录包含 `judge_results`、`judge_votes`、`majority_vote`、`agreement_count` 和 `unanimous`。若部分裁判调用失败，成功结果仍会保存，续跑时只补缺失裁判。
