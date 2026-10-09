# Runtime screenshots used in Figure 3

- A: `openrec_rec_items.png`: item_114 card from user_0 / scene_0 / default.
- B: `openrec_node_config.png`: operation presets and complete visible configuration
  JSON. LaTeX excludes the graph canvas and class metadata to give the policy fields
  more space. operationName selects WeightedChannelOperationRule in this capture;
  the presence of diversity parameters does not imply that policy was executed.
- C: `openrec_feature.png`: the entire supplied user/candidate feature selection,
  including attribute, content and statistical groups. No feature columns are cropped.

The layout stacks A and C on the left and places B on the right. Original PNG pixels
are unchanged; English headings and viewports are defined in main.tex.
Only the three manuscript screenshots are retained and included in the source ZIP.
No additional screenshot is requested. Runtime validation uses text/API evidence.

# Structural diagrams

Figure 1 uses `cluster-architecture.tex`, adapted from the organization profile.
Figure 2 uses `item-graph.tex`, verified against all 17 nodes / 18 dependencies
in the actual serving configuration. See DIAGRAMS.zh-CN.md and item-graph-source.json.
Both are vector diagrams, included by main.tex without rasterization.
