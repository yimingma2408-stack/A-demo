## A-share Multi-factor Project Demo

这是一个基础但完整的 A 股多因子研究项目，覆盖：

1. 数据读取与清洗
2. 数据质量检查与基础分析
3. 价格量价因子构建
4. 因子 IC 与分组收益检验
5. 多因子合成
6. 简单多空组合回测
7. Notebook 示例复现

项目默认使用本地 `data/raw/stocks_panel_daily_qfq_baostock.csv`，因此核心示例可以离线运行。

## 目录结构

```text
.
├── data/
│   ├── raw/                       # 原始行情数据
│   └── processed/                 # 清洗、因子、回测结果
├── notebook/
│   └── quant_project_demo.ipynb   # 完整示例 notebook
├── scripts/
│   ├── data_processing.py         # 清洗与质量报告
│   ├── analysis.py                # 基础统计分析
│   ├── factors.py                 # 因子构建与标准化
│   ├── factor_evaluation.py       # IC、分组收益、因子合成
│   ├── backtest.py                # 多空回测
│   └── pipeline.py                # 一键运行完整流程
└── requirements.txt
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行完整 pipeline：

```bash
python -m scripts.pipeline
```

运行后会在 `data/processed/` 生成：

- `clean_daily_data.csv`
- `factor_panel.csv`
- `factor_ic_daily.csv`
- `factor_ic_summary.csv`
- `factor_panel_scored.csv`
- `backtest_nav.csv`

## Notebook 示例

打开：

```bash
jupyter lab notebook/quant_project_demo.ipynb
```

Notebook 会演示从原始数据开始完成清洗、分析、因子挖掘、检验和回测。

## 研究说明

当前因子全部来自日频价格与成交量数据，包括反转、动量、低波、流动性、换手、趋势、振幅、量比、价格位置和成交额规模等。回测采用日频 close-to-close 的多空组合：在 t 日用因子打分选股，使用 t+1 日收益计算组合表现，避免直接使用未来收益。

本项目用于研究流程演示，不构成投资建议。真实投资前还需要处理停复牌、涨跌停、交易成本、滑点、行业/市值中性化、成分股变迁和幸存者偏差等问题。
