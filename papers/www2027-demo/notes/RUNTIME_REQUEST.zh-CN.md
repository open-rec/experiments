# 新版 Demo 运行材料清单

本轮主线改为推荐策略、特征评分、反馈路径。用户已确认 cluster 环境启动完成。以下为进一步排练与取证的建议，不是新增开发要求，也不是已有实验结果；正文使用现有 cluster 页面和功能进行演示。使用独立演示环境和例子身份；无需生产流量或重跑全部算法实验。

## 1. 基础资料

作者信息已确认：Zheng Ma / Alibaba / Hangzhou, China / xsank@foxmail.com。现场排练需记录：公开仓库版本；完整 cluster 主机 CPU、RAM、磁盘、OS、部署拓扑；用户/物品/事件规模；所用 graph、active recall、model 与 sidecar。训练提前完成，现场不等待长任务。standalone 只作部分流程备用。

## 2. 图片已齐，其他状态使用文字记录

按作者选择，正文只使用现有三张截图，不再要求 operation 或在线部署页截图：

- `openrec_rec_items.png`：推荐卡片、通道、rank 分数和反馈入口。
- `openrec_node_config.png`：operation 节点的策略预设及 JSON 参数。
- `openrec_feature.png`：用户和候选两列特征选择，保留提供截图的完整内容。

正文已文字介绍 operation 的通道配额/滑窗多样性，以及特征选择、模型激活和实际加载核对。后续只需用文字或原始 JSON 补充实际运行状态：

| 材料 | 所需字段 |
|---|---|
| operation 配置 | 实验/图版本、operationName、channelRatios 或 windowSize/repeatK/diversityDimensions、规则加载确认 |
| 模型运行信息 | target、模型类型/版本、选定特征/维度、sidecar 标识、console record 和 rank-engine loaded state |
| 反馈验证记录 | 原始事件、ACK、规则投影、匹配请求与返回值 |

无需将这些 JSON 再截成图片。机制说明与实际运行验证仍须区分：文字不能将尚未测量或核对的状态写成已完成结果。

## 3. 任务 A：候选供给与列表策略

1. 固定用户、scene、请求大小与 active artifacts。保存 baseline 图和 debug 响应。
2. 选择确实贡献候选的召回节点，禁用并发布；记录 ACK、目标路由 ready 和实际选中图。
3. 确认对应诊断为 `DISABLED`。Top-K 不变并不代表失败，其他通道可能召回同一候选。
4. 恢复通道；准备含重复类目/标签的候选数据，通过现有预设比较 `WeightedChannelOperationRule` 与 `SlidingWindowDiversityOperationRule`。
5. 确認插件/规则加载成功。记录策略参数、原始列表、主通道、类目/标签、返回长度，核对滑窗限制。严格约束导致结果变少应如实展示。
6. 使用受控 API 请求避免 Web 自动曝光影响策略比较，或完整记录该变化。不同请求并未共享固定候选快照，不能将结果包装成严格因果实验。最后恢复原图。

## 4. 任务 B：特征选择到在线评分

基本展示使用当前已加载模型，核对其特征、sidecar 和推荐分数即可。以下两版本切换/恢复流程用于选择展示模型生命周期时，不是全部必须执行的投稿条件。

1. 准备两份可部署且已评估的 LR/FM/LightGBM release，保存原始 sidecar。
2. 展示选定特征、定义、拟合表示和输入维度；区分逻辑特征数量与编码维度。
3. 展示对应在线原始字段或 feature snapshot，可用只读 API/数据检查器。不能以 catalog 的 online 标记替代实际物化检查。
4. 发布第二份模型，分别记录 console ACK 和 rank-engine 实际 loaded state；真实推荐路径必须有 rank score。
5. 记录模型持有的特征快照/刷新配置。改变 feature selection 需要重新训练，不修改现有权重解释。
6. 恢复保留模型及原编码；模型对 item/user target 全局生效，不声称每个 graph experiment 各自独立模型。

## 5. 任务 C：反馈影响下一次推荐

1. 使用另一份准备好的测试身份，放在最后演示。核对 dislike 支持且目标物品确实在候选中。
2. 从 Web 提交物品级 dislike；保存原始请求、ACK、实际 Redis dislike 成员及用户/物品 feature snapshot。
3. 等待异步投影完成再验证推荐。返回列表缺少目标物品只能作为结果，不能单独归因于 dislike。
4. 准备两份除 dislike 过滤开关外相同的图，均关闭曝光过滤，并确认 cluster 的 `collector.fake-expose.enabled=false`；用不会额外上报曝光的 API 客户端进行匹配请求，固定用户、scene、触发项、请求大小与 artifacts。先确认目标物品在关闭 dislike 的请求中可进入列表，再核对开启后的排除。若候选不足或持续变化使对照失效，记录失败并重排 fixture，不伪称验证成功。
5. 核对实际图还可能应用的其他过滤/operation 规则。图的恢复不删除用户反馈，Web reset 也不代表历史库或模型快照全局回放。
6. 若展示模型输入更新，记录 Redis 投影时间与 rank-engine 强制/计划缓存刷新时间，二者分开。可用 `/model/refresh-features`，记录具体 target 和结果。不承诺分数/排序必变或自动在线训练。

可选扩展：click 进入召回 trigger、类目/标签级 dislike；它们不必都塞进四页。

## 6. 可选性能测量（正文当前不作性能主张）

每次操作保存请求、响应、实际状态核对与失败原因。建议：请求预热 100 次后测 1,000 次；状态变更分别尝试 20 次。这些是建议协议，不是会议规定。不足次数则报告实际次数与范围。

- 推荐延迟：普通/debug 分开；固定工作负载与并发，报告失败数。
- 图与模型：从提交到目标路由 ready/模型实际 loaded 的时间，并记录 ACK；回滚单独记录。
- 反馈：提交到规则投影、特征快照更新、编码缓存刷新的时间分别报告，注明轮询间隔和计时边界。
- 成功标准：报告验证通过数/尝试总数、超时和失败；成功操作耗时单独注明，不能排除失败后只报漂亮延迟。
- 列表策略：保留原始结果与属性核对；这是功能证据，不是推荐质量评估或满意度研究。

填写 `runtime_measurements.csv`，所有空白表示未测。`raw_evidence_path` 指向逐次原始数据；分时读取的跨服务状态不是全局原子快照。

## 7. 可选视频脚本（建议 5 分钟）

| 时段 | 内容 |
|---|---|
| 0:00–0:30 | Web 推荐页面与核心问题：离线组件如何进入真实推荐流程 |
| 0:30–1:50 | 候选供给、禁用通道、列表策略比较 |
| 1:50–3:10 | 选定特征、fitted sidecar、loaded 模型与 rank score |
| 3:10–4:30 | dislike、异步投影、匹配请求与后续过滤 |
| 4:30–5:00 | 源码/复现入口与主要能力边界 |

使用真实录屏；跳过等待要标记剪辑，不以剪辑暗示耗时。现有界面可配英文注释。

## 8. 备用材料与既有验收

召回 staging/active alias 切换、模型/图回滚可作为追问时的备用操作。保存 alias 实际指向和恢复结果，不只截 toast。

`example_cluster/verify_daily_recall.sh`、`verify_rank_model.sh`、`verify_feature_parity_e2e.sh` 是已有验收入口。本轮只读取，没有执行；模型验收脚本会写数据并重启服务。已有记录不能替代新版三个任务的排练和原始请求/状态记录。
