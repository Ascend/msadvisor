# 改图组件介绍

## 简介

graph_refactor 是 AutoOptimizer 工具的一个基础组件，提供简易的改图接口，解决用户改图难度大、学习成本高的问题。目前支持 onnx 模型的以下改图功能：

- [x] 加载和保存模型
- [x] 查询和修改单个节点信息
- [x] 新增节点，根据条件插入节点
- [x] 删除指定节点

## 使用方法

- BaseNode 使用方法参见 [BaseNode 说明](../../docs/graph_refactor_BaseNode.md)
- BaseGraph 使用方法参见 [BaseGraph 说明](../../docs/graph_refactor_BaseGraph.md)
- 接口详见 [API 说明和示例](../../docs/graph_refactor_API.md)