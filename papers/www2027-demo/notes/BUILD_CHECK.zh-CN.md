# 草稿校验记录

日期：2026-10-09。整体投稿审查见 SUBMISSION_REVIEW.zh-CN.md；已修正点击任务说明、特征分组与 LogLoss 解释并重新编译。experiments 参考文献改用 v0.1.0 标签和 Version 0.1.0；本地标签下的报告与所用报告内容一致（本轮投稿检查已确认两个引用的公开原始文件均返回 HTTP 200）。本轮将 Interactive Demonstration 改写为现有 cluster 部署的交互说明，明确已有启动入口、推荐页和控制台，以及现场准备方式；不再将详细排练协议写入正文。

- 图 1：依据 .github/profile/README.md 重绘的 Cluster architecture；图 2：依据实际 item_graph.json 的分组 DAG。两图在第 2 页系统设计部分并排呈现，使用 TikZ 矢量文字与连线。
- 图 3：原有三张真实界面图，位于第 3 页，图片内容和布局保留。
- 节点/边验证：从图 2 源码提取节点与组间箭头，展开后与 JSON 精确匹配，共 17 节点和 18 依赖；记录见 ../figures/item-graph-source.json。
- README 的图未列 query_emb/sparse，论文图以实际配置补齐。user_feature 仅传递 user ID，item_feature 为预留 hook，图注不夸大其实现。
- Evaluation 新增 Popularity/LR/FM/LightGBM 结果表；表内均值和正文 Random-policy AUC 样本标准差均与逐次 CSV 核对。Tectonic 0.17.0 编译为四页，含参考文献；正文不存在 TO COMPLETE 标记。
- 已渲染检查四页，无横纵向溢出、未定义引用或 BibTeX 警告；保留普通 underfull 提示。
- 未修改产品、上游 README 或运行服务；未新增训练、性能测量或排练结果。cluster 已运行依据用户确认，当前工具未重新启动或执行现场交互验收。
- 源码包仅包含 main.tex、references.bib、三张正文截图和两份矢量图 .tex。图源核对记录保留在 figures/，审查和证据记录集中在 notes/。作者信息已更新为 Zheng Ma / Alibaba / Hangzhou, China / xsank@foxmail.com；现场部署资料与交互任务仍需排练核对；视频和性能测量为可选补充，正文不再包含占位。

PDF SHA-256：`47c721f2e4ab506295347232cd3201982c711316321ce8641f376dd8da32f6d9`。

目录整理后检查：精简源码包在独立临时目录编译成功，四页文本与渲染像素和当前 PDF 完全一致；内部准备说明已区分基本展示与可选扩展。
