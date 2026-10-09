# OpenRec WWW 2027 Demo

论文：**OpenRec: Bridging Offline Experiments and Interactive Web Recommendation**。

作者：Zheng Ma · Alibaba · Hangzhou, China · xsank@foxmail.com。

当前为四页英文投稿稿，含参考文献，无待补占位。系统展示使用现有 cluster 模式，围绕候选与列表策略、特征与评分、反馈与后续请求展开。KuaiRand 结果来自已有 v0.1.0 实验报告，没有新增测量或线上效果声明。

## 论文文件

- [main.pdf](main.pdf)：当前四页 PDF。
- [main.tex](main.tex)、[references.bib](references.bib)：论文与六项实际引用。
- [figures/](figures/)：两张 TikZ 系统图、三张正文截图及图源说明。
- [openrec-demo-source.zip](openrec-demo-source.zip)：可上传 Overleaf 的精简编译源码包，主文件为 `main.tex`。

## 内部支撑材料

[notes/](notes/) 保存投稿核对与证据，不属于论文正文，也不放入 Overleaf 源码包：

- [投稿审查](notes/SUBMISSION_REVIEW.zh-CN.md)、[编译检查](notes/BUILD_CHECK.zh-CN.md)。
- [核心价值](notes/CORE_VALUE.zh-CN.md)、[实现与实验依据](notes/EVIDENCE.zh-CN.md)、[来源快照](notes/source_snapshot.json)。
- [现场准备](notes/RUNTIME_REQUEST.zh-CN.md)、[可选测量模板](notes/runtime_measurements.csv)。测量空白表示未测，不是结果。

## 编译与投稿

在本目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# 或
tectonic --keep-logs main.tex
```

[WWW 2027 Demo CFP](https://www2027.thewebconf.org/demos/) 要求英文 ACM 双栏、最多四页含参考文献、非匿名作者信息、伦理说明和现场部署安排。截止时间为 **2026-11-16 AoE**，提交前以官网最新要求为准。视频为鼓励提供的补充材料，不是必交项。

项目首页：https://github.com/open-rec。
