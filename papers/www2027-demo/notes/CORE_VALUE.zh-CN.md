# OpenRec 核心价值与 WWW Demo 定位

核对日期：2026-10-09。范围：十个主要子项目的文档、代表性实现、跨组件契约与既有验证记录。没有声称逐行审计全部代码，也没有重新运行服务或性能实验。结论对应当前源码与 0.1.0 兼容组件集合；文件摘要记录于 `source_snapshot.json`。

## 建议定位

**OpenRec 将离线推荐算法与特征实验接入可交互的 Web 推荐流程，使开发者能够观察候选供给、列表策略、模型输入和用户反馈之间的联系。**

建议标题：**OpenRec: Bridging Offline Experiments and Interactive Web Recommendation**。

核心价值是把推荐中通常分开研究和实现的环节，通过明确的数据、特征和模型契约连接起来，并让这些连接能通过真实请求被检查。本文不把算法数量、基础设施数量、版本管理或 AUC 数字作为主要贡献，也不声称这些机制是首次提出。

## 子项目核对与论文取舍

| 子项目 | 核对的代表性实现（相对于工作区） | 对核心价值的贡献与边界 |
|---|---|---|
| rec-algorithm | `algorithm/recall/item_cf_i2i.py`、`bm25.py`；`algorithm/feature/point_in_time.py`、`feature_space.py`；`jobs/spark/point_in_time.py`、`rank_job.py`、`io.py` | 候选算法、时间一致的训练样本和可供线上导入的编码契约。研究代码支持不等于 cluster 能训练或部署全部配置。 |
| data-processor | `feature-core/.../FeatureUpdates.java`、`EventFeatureAccumulator.java`；共享 parity fixture；Flink/Spark 文档与发布记录 | 一次交互进入用户/物品聚合、服务投影及历史数据；两个流引擎是可替换实现，不是双活一致性保证。 |
| rank-engine | `service/feature_service.py`、`server.py` 的加载、`_refresh_features` 与评分路径 | 使用持久化的 fitted FeatureSpace，不在线重拟合；维护模型与特征快照配对。Redis 更新与编码缓存刷新是两步，不能承诺零延迟。 |
| rec-server | `GraphPlan.java`；`TriggerNode`、`FilterNode`、`BlackNode`、`CombineNode`、`RankScoreFusion`；contrib 的 `SlidingWindowDiversityOperationRule` | 连接多路召回、曝光/负反馈过滤、排序融合与列表策略。候选供给与结果列表须区分；不存在全候选因果解释。 |
| rec-console | `model_training.py`、`model_releases.py`、`serving_graphs.py`；前端 `ModelPage.tsx`、`main.tsx` | 可选支持的特征、提交训练、检查产物、改变图策略并显式部署。是推荐实验的操作界面，不是论文唯一主角。 |
| sdk | Java/Go/Python API 映射；`python-client/openrec/client.py`；Java 协议依赖 | 将应用请求与交互送入同一服务契约。Go/Python 类型化结果目前没有完整召回诊断，演示用原始 HTTP JSON。 |
| bigdata-platform | `platform.sh` 依赖闭包、README 的模式和运行边界 | 为别人复现集成环境提供支撑，不把 Docker 组件清单当创新或生产级可用性证明。 |
| model | `feature/catalog/README.md`、catalog、拟合 sidecar 布局、默认产物校验说明 | 区分逻辑语义、模型选择、拟合编码、实际数值；默认 artifact 不是任意数据集的通用最优模型。 |
| example | `release/openrec.json`、`VALIDATION.md`；Web 的 `FeedbackService` 与 `app.js`；`verify_feature_parity_e2e.sh` | 真正可交互的浏览器入口、反馈路径、版本组装和跨组件验收。热门/新品快捷页不能用于证明 rank-engine 执行。 |
| experiments | `openrec_experiments/runner.py`、`evaluation.py`、数据适配目录及结果报告 | 复用算法/编码并记录可追溯实验。EB-NeRD、KuaiRand、Synerise、Music-CRS 展示研究适配范围，不代表四个领域均已有在线演示或官方榜单结果。 |

`rec-server` 的 graph/proto/contrib 分别提供通用 DAG、应用协议、列表规则扩展点。它们支持扩展推荐流程，但添加节点、特征或模型仍需要工程实现，不是无代码接入任意算法。

## 三项主要贡献及对应现场任务

### 1. 从候选供给到最终列表的可组合推荐

- 机制：并行召回、候选合并与过滤、召回/排序分数融合、通道配额或滑窗多样性规则。
- 场景：先禁用一个有供给的通道，观察诊断；恢复后比较通道配额与类目多样性规则。
- 证据：过滤前候选数、通道状态、实际策略、结果通道/类目/长度。
- 边界：通道重叠可能使 Top-K 不变；配额按主通道归属；多样性约束可能减少返回数量；缺失 operation 插件可能回退，所以先确认规则加载。

### 2. 从离线特征选择到在线打分的契约连续性

- 机制：canonical catalog → supported selection → fitted sidecar → online materialization → score。
- 场景：打开已完成训练的模型版本，查看特征选择与维度，联系在线来源，部署后验证 loaded state 与实际 rank score。
- 证据：原始模型/sidecar、定义指纹、选定特征、缓存更新时间和真实推荐响应。
- 边界：共享定义及 fixture 不等于任意部署、所有实体、所有时间点都一致；新增特征需要生产端实现和历史准备；切换特征必须重新训练。

### 3. 从 Web 反馈到后续请求和离线历史的数据路径

- 机制：Web → ingestion → Kafka → 流处理 → Redis 与历史；click 供 trigger、expose/dislike 供过滤；行为聚合供特征，历史供未来训练/召回任务。
- 场景：提交物品级 dislike，等待投影，比较配置匹配的请求；补充展示特征快照更新。
- 证据：反馈 ACK、规则实际存在、输入候选、过滤开关、后续响应、特征快照。
- 边界：原列表消失不能单独证明 dislike，因为自动曝光、候选变化也可能导致消失；需固定请求并控制曝光过滤。不会立即自动训练或自动发布模型。

## 为什么与 WWW 相关

[WWW 2027 Demo CFP](https://www2027.thewebconf.org/demos/) 要求已实现并测试的系统，强调参会者能直接交互、对 Web 社区主题的关联及现场部署方案。上述任务围绕 Web 内容推荐、列表呈现和用户反馈，能够让参会者操作并检查实际行为；这比单独展示后台发布与回滚更能说明应用价值。这是我们的投稿定位判断，不是录用保证。

同赛道 [RecAI](https://arxiv.org/html/2403.06465v1) 为“围绕推荐应用能力介绍工具系统”提供参考。不要据此声称其他推荐库完全没有工程、上线或反馈能力；本文没有做穷尽性竞品评测。

## 当前证据与应补证据

已有：源码实现、0.1.0 发布检查、跨组件特征契约与共享 fixture、历史 cluster 验收、KuaiRand 本地离线结果。

已补：三张作者提供的运行界面截图，分别展示推荐卡片、operation 节点策略配置和用户/候选特征选择。

现场准备建议：走通上述三项交互，并核对实际策略、loaded 模型与反馈投影。投影/缓存延迟测量是可选增强，不是正文待补项或投稿前置条件。用户已确认 cluster 启动完成，本次文档审查没有重新执行完整演示；论文不宣称用户研究、线上质量提升或生产规模性能结论。

README/代码注释有历史残留：例如 FeedbackService 顶部还描述较早的反馈范围，而当前方法、前端、BlackNode/CombineNode 已实现 dislike 路径。论文以相互核对的调用路径为依据，不从单一说明推断功能。
