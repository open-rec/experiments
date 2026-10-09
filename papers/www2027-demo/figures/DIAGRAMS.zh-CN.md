# 论文结构图来源与核对

## 图 1：Cluster architecture

来源为工作区 `.github/profile/README.md` 的 Cluster architecture Mermaid，以及当前分布式组件的实现边界。论文使用 `cluster-architecture.tex` 重新排为英文矢量图，保留 Web/SDK、rec-server、Redis/Elasticsearch、rank-engine、Kafka、data-processor、历史存储、rec-algorithm、模型产物、Airflow/runner、rec-console、Prometheus/Grafana。

为适应四页，将服务存储、历史存储和调度/runner 分组，画出主要 API/数据路径与部分控制关系。实线混合表示图中标明角色之间的 API/数据交互，不是统一方向的物理数据拷贝；虚线表示控制台控制或访问。图下说明被省略的 Airflow 控制、metrics scrape，以及 staging/alias 的区别。profile 的默认流实现为 Spark；源码支持 Flink 替代，所以图中注明 Spark or Flink，不表示两个引擎同时运行。

该图是依据源码结构制作的架构说明，不是当前主机全部容器已健康的运行证据。

## 图 2：Item graph

主要依据 `rec-server/server/src/main/resources/item_graph.json`。rec-server README 的 Mermaid 少列 `query_emb` 和 `sparse`，论文采用配置中的完整节点集合，不修改上游仓库。

`item-graph.tex` 将依赖相同的节点分组。节点名保持原样，箭头展开到组内所有节点。已程序核对：17 个节点全部出现；8 条组间连线展开后，精确对应 JSON 的 18 条依赖，没有新增或遗漏。分组与源文件 SHA-256 记录于 `item-graph-source.json`。

必须区分节点名称与实际实现：

- `user_feature` 当前输出 user ID；
- `item_feature` 当前为 reserved hook；
- 特征值加载、拟合编码复用和缓存快照在 rank-engine 中完成。

因此图 2 展示调度依赖，不把两个 feature 节点画成已经执行了全部特征工程。filter/black 为 combine 提供过滤输入，不能误画成召回后依次执行的独立串行阶段。

## 图 3：真实运行界面

保留三张现有截图的布局与内容：推荐卡片、operation 策略/JSON、用户与候选特征选择。新增结构图不替代运行截图，也不增加截图采集要求。

两张结构图源码由 main.tex 的 input 引用，编译时仍为矢量文字与连线；源码包必须包含这两个 .tex 文件。
