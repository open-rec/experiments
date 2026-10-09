# 论文主张与实现依据

核对时间：2026-10-09。路径均相对 OpenRec 工作区根目录。本次是只读代码/文档审查与论文编译，没有启动或变更服务，没有新跑模型实验。

## 本轮新增的推荐能力依据

完整子项目审查见 `CORE_VALUE.zh-CN.md`。下列机制成为正文主线，原来的发布与回滚作为支撑。

| 主张 | 依据（相对工作区） | 边界 |
|---|---|---|
| 分数融合与列表策略 | `rec-server/server/src/main/java/com/openrec/graph/node/RankScoreFusion.java`；`rec-server/contrib/src/main/java/com/openrec/contrib/operation/impl/SlidingWindowDiversityOperationRule.java`；`rec-console/frontend/src/main.tsx` | 主通道配额不等于全部来源覆盖；多样性可缩短列表；检查插件加载 |
| 逻辑特征与拟合表示分离 | `model/feature/catalog/README.md`；`rec-algorithm/algorithm/feature/feature_space.py` | catalog 支持不等于数据已物化 |
| 时间一致的训练准备 | `rec-algorithm/algorithm/feature/point_in_time.py`、`jobs/spark/point_in_time.py`、`jobs/spark/rank_job.py` | 按标签时刻与 mutation 可见性处理，不泛化为所有实验绝无泄漏 |
| 特征选择与离线训练 | `rec-console/rec_console/model_training.py`；`rec-console/frontend/src/ModelPage.tsx` | 可训练特征是支持子集，研究构件不自动可部署 |
| 模型/编码快照配对 | `rank-engine/server.py`；`rank-engine/service/feature_service.py` | 流更新和缓存刷新分离；失败可保留旧快照 |
| 反馈影响后续请求 | `example/web/src/main/resources/static/js/app.js`；Web `FeedbackService.java`；rec-server 的 `TriggerNode`、`FilterNode`、`BlackNode`、`CombineNode` | ACK 不等于投影生效；列表变化不能单独归因 |
| 行为特征与历史 | `data-processor/feature-core/src/main/java/com/openrec/dp/feature/FeatureUpdates.java`、`EventFeatureAccumulator.java`；`rec-algorithm/jobs/spark/io.py` | 不即时自动训练；流引擎是替代方案，不是双活保证 |
| 共享契约的验证 | `example/example_cluster/verify_feature_parity_e2e.sh`；`example/release/VALIDATION.md`；共享 `event-feature-parity.json` | 既有 fixture/历史验收，不是本轮新跑的全覆盖验证 |

## 核心证据表

| 论文主张 | 依据 | 应保留的边界 |
|---|---|---|
| 0.1.0 为协调源码发布 | `example/release/openrec.json`、`example/RELEASE_NOTES.md` | 源码发布不等于已发布所有容器镜像 |
| UI 可编辑节点并发布实验图、保留回滚历史 | `rec-console/frontend/src/main.tsx` 的 Serving Graph 模块；`rec-console/rec_console/serving_graphs.py` | 发布整个 snapshot；新 Java 节点仍需开发 |
| 请求选取图快照并按实验路由 | `rec-server/server/src/main/java/com/openrec/ab/AbExperimentService.java` | 显式 `params.ab` 不等同于完整统计 A/B 实验平台 |
| 编译校验后激活图 | `rec-server/server/src/main/java/com/openrec/service/rec/ServingGraphService.java` 与 `RecService.java` | 不保证业务效果提升 |
| readiness 与 liveness 分开 | `rec-server/server/src/main/java/com/openrec/controller/api/RecommendController.java`、`service/rec/RecommendationReadiness.java`；`example/docs/recommendation-readiness.md` | warmup 验证代表请求，不是所有路径/延迟的数学保证 |
| recallDiagnostics 为过滤前状态和数量 | `rec-server/server/src/main/java/com/openrec/service/rec/RecService.java` 的 `recallDiagnostics()`；`RecallDiagnosticsTest.java` | 不是逐候选完整 lineage；失败/超时不得算成功供给 |
| 卡片有召回通道和 recall/rank 分数 | `example/web/src/main/resources/static/index.html`、`static/js/app.js` | UI 卡片与 debug JSON 是两个证据渠道；不声称现成统一 trace dashboard |
| 曝光会改变后续结果 | 同上，及 Web 事件上报逻辑 | cluster 真实可见曝光；standalone 可产生伪曝光；比较前控制反馈 |
| 召回 staging/发布/切换/回滚 | `rec-console/rec_console/recall_indexes.py`；`rec-console/test/integration/test_recall_lifecycle.py` | 写 staging 不会自动生效；published marker 与 active alias 同次操作 |
| 模型校验和显式加载 | `rec-console/rec_console/model_releases.py`；`frontend/src/ModelPage.tsx` | 校验 weights/sidecar/model type/dimension；训练与激活分开 |
| LR/FM/LightGBM 可服务 | `rank-engine/RELEASE_NOTES.md`、`rec-console/README.md` | 不能据此说全部实验特征和 Transformer 都可服务 |
| 全局 target 模型 | `rec-console/README.md` Global features and model lifecycle | item/user 各自全局激活，不能宣传不同图实验各有独立模型版本 |
| 多组件版本分别管理 | 图、召回、模型管理类分别执行 | 没有 graph+recall+model 跨服务原子事务 |
| standalone/cluster 能力不同 | `example/RELEASE_NOTES.md`、`rec-console/RELEASE_NOTES.md` | 全流程演示需要 cluster；standalone 通常跳过模型排序 |
| 已有验证记录 | `example/release/VALIDATION.md` | 明确是 2026-10-05 发布记录/历史集成证据，不是本次新测 |
| Console 尚无认证/RBAC | `rec-console/RELEASE_NOTES.md` | 现场隔离，不直接公开管理写接口 |

## 数字来源

唯一写入正文的算法数字来自：

- `experiments/results/kuairand-1k-pit-global-v1.md`
- `experiments/results/kuairand-1k-pit-global-v1.csv`
- `experiments/studies/baselines/kuairand-pit-global.json`

起始实验仓库 HEAD：`dc4c62e2d418a47fda778bd6e84c958f6792da53`。
该 revision 是本次引用结果文件的仓库状态，不替代每个原始 run manifest 的历史代码版本。完整运行产物位于忽略目录时，公开报告不等于原始全部产物已公开。

协议摘要：`f7a44d1222b70a7ae6064f93c1974c1397a149150edd0d42349707865de09220`。
学习模型报告三个随机种子的均值和样本标准差；Popularity 单次确定性结果。Random-policy 样本不进入训练、验证选择或特征历史；评估反馈冻结并有一小时延迟。

为什么不写 EB-NeRD 的最高分：`ebnerd-small-common-features-v2.md` 明确有 112 个实验性 offline-only 逻辑特征。论文不把这些特征写成当前 demo 的线上能力。

## 尚不能声称的结果

- 正式在线 A/B uplift、生产流量规模、QPS、p95/p99 SLA。
- “所有 online/offline 特征严格一致”或“任意模型一键上线”。
- 失败发布在所有网络/持久化故障下具有全局原子性。
- 图 rollback 自动恢复原推荐列表、所有反馈或特征库状态。
- 现有 Web 页面已显示所有 debug trace 及图/model/alias 的统一 provenance。
- 此次已经完成三个任务的现场排练、截图采集或用户研究。
- 新增英文 UI、权限系统、端到端回放页面已实现。

## 最低补证闭环

对每个交互任务保存操作前状态、操作请求/响应、操作后状态，以及一条完整推荐请求/响应。图发布后核对 readiness 与实际策略；模型核对选定特征、实际加载版本和 rank score；反馈核对投影与控制曝光过滤的匹配请求。备用召回演示核对 alias 实际指向。不能仅截取“发布成功”的 toast。

论文已移除 runtime 红色待补段落，并以已有 KuaiRand 结果及发布验证记录支撑 Evaluation。尚未采集的延迟/吞吐量不作为正文主张。排练仍须验证交互任务；若某个任务无法稳定完成，应删改其描述，不能把预期行为改成完成事实。

## 本轮作者提供的运行截图

正文图 3 使用 `openrec_rec_items.png` 的推荐卡片、`openrec_node_config.png` 的 operation 策略与 JSON 参数区域，以及完整的 `openrec_feature.png`。后两张替换此前的 rank 节点字段和局部用户属性图；不增加子图数量。原 PNG 均未改动，布局和局部显示由 LaTeX 完成。

operationName 显示 WeightedChannelOperationRule；同一 JSON 出现 diversityDimensions 不代表滑窗多样性策略已执行。特征图包含用户与候选两列及属性、内容、统计特征分组，但仍是选择界面，不证明训练完成或实际模型加载。推荐卡片显示 rank 值，不据此推断当前模型类型。

这些截图是界面与显示状态证据，不是任务前后对照、特征一致性或性能证据。目录仅保留正文使用的三张截图。

## Cluster 与 Item DAG 图的依据

图 1 参考 `.github/profile/README.md`，以存储/调度分组呈现主要链路；图 2 以实际 `item_graph.json` 为准，补入 README Mermaid 未列的 query_emb 和 sparse。分组图已验证完整覆盖 17 节点和 18 依赖。user_feature 当前提供 user ID，item_feature 为预留 hook；图注明确实际编码在 rank-engine，不能从节点名字推断额外能力。详见 `figures/DIAGRAMS.zh-CN.md`。
