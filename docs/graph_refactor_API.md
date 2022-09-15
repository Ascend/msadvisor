# API 说明和示例

## BaseGraph API

### 查询节点

**g[name] -> Union[Node, Initializer, PlaceHolder]** 

- 根据节点名称查询单个节点，返回该节点。
- `name(str)` - 节点名称

**get_nodes(op_type) -> Union[List[Node], List[PlaceHolder], List[Initializer]]** 

- 根据节点类型名称查询节点，返回由该类型节点组成的列表。
- `op_type(str)` - 节点类型名称，如 `'Add'` 。

**get_prev_node(input_name) -> Node**

- 查询某条边的前驱节点并返回该节点。若不存在返回 `None` 。
- `input_name(str)` - 图中存在的边的名称

**get_next_nodes(output_name) -> List[Node]**

- 查询某条边的后继节点列表并返回该列表。若不存在返回 `[]` 。
- `output_name(str)` - 图中存在的边的名称

<details>
  <summary> sample code </summary>

```python
# 加载模型
g = OnnxGraph.parse('model.onnx') 

# 获取名称为'Add_0'的节点
node = g['Add_0'] 

# 获取所有Add类型节点
nodes = g.get_nodes('Add') 

# 获取 node 的所有前驱节点
prev_nodes = []
for i in node.inputs:
	prev_nodes.append(g.get_prev_node(i)) 

# 获取 node 的所有后继节点
next_nodes = []
for o in node.outputs:
	next_nodes.extend(g.get_next_nodes(o))
next_nodes = list(set(next_nodes))
```

</details>

### 修改节点

**g[name] = new_node**

- 替换图中单个节点。支持以下场景：
  - 算子节点替换成输入输出相同的算子节点
  - 常量节点替换成输入节点
  - 算子节点替换成输入节点

- `name(str)` - 待替换节点名称

### 添加节点

**add_input(name, dtype, shape) -> PlaceHolder**

- 添加整网输入节点。
- `name(str)` - 输入节点名称 \
  `dtype(str)` - 输入数据类型 \
  `shape(List[int])` - 输入数据形状

**add_output(name, dtype, shape) -> PlaceHolder**

- 添加整网输出节点。
- `name(str)` - 输出节点名称 \
  `dtype(str)` - 输出数据类型 \
  `shape(List[int])` - 输出维度信息

**add_initializer(name, value) -> Initializer**

- 添加常量节点。
- `name(str)` - 常量节点名称 \
  `value(np.ndarray)` - 常量值

**add_node(name, op_type, attrs=None) -> Node:**

- 添加孤立的算子节点。
- `name(str)` - 算子节点名称 \
  `op_type(str)` - 算子类型。参见 [Onnx 算子标准 IR](https://github.com/onnx/onnx/blob/main/docs/Operators.md) 。\
  `attrs(Dict[str, Object])` - 算子属性。参见 [Onnx 算子标准 IR](https://github.com/onnx/onnx/blob/main/docs/Operators.md) 。

<details>
  <summary> sample code </summary>

```python
# 添加整网输入输出节点
new_input = g.add_input('new_input', 'float32', [1,3,224,224])
new_output = g.add_output('new_output', 'float32', [1,3,224,224])

# 添加常量节点
new_ini = g.add_initializer('new_ini', np.array([1,1,1]))

# 添加算子节点
new_op = g.add_node('Transpose_new', 'Transpose', {'perm':[1,0,2]})
```

</details>

### 插入节点

**insert_node(refer_name, insert_node, refer_index=0, mode='after')**

- 在参考节点的指定输出边（或输入边）插入单输入单输出节点
- `refer_name(str)` - 参考节点的名称 \
  `insert_node(Node)` - 插入节点 \
  `refer_index(int)` - 对于 after 模式， 用于指定参考节点的输出边；对于 before 模式，用于指定参考节点的输入边 \
  `mode(str)` - after 模式表示将节点插入到参考节点后；before 模式表示将节点插入到参考节点前。默认为 after

**connect_node(insert_node, prev_nodes_info, next_nodes_info)**

- 将节点插入图中指定位置，自动连边。
- `insert_node(Node)` - 待插入节点 \
  `prev_nodes_info(List[str])` - 指定前驱节点，列表中的每个字符串对应待插入节点的一个输入 \
  `next_nodes_info(List[str])` - 指定后继节点，列表中的每个字符串对应待插入节点的一个输出 

<details>
  <summary> sample code </summary>

```python
# 添加并插入单输入单输出算子
new_cast_0 = g.add_node('new_cast_0', 'Cast', {'to':6}) 
new_cast_1 = g.add_node('new_cast_1', 'Cast', {'to':6}) 
g.insert_node('reference_node', new_cast_0) # 在参考节点后插入Cast算子
g.insert_node('reference_node', new_cast_1, 1, 'before') # 在参考节点的第1条输入边插入Cast算子

# 添加并插入多输入多输出算子
# 前驱节点为单输出，后继节点为单输入时可以简写，比如'Add_0:0'写成'Add_0'
split_ini = g.add_initializer('split_ini', np.array([1,1,1]))
new_split = g.add_node('new_split', 'Split')
g.connect_node(
                new_split,
                ['Add_0', 'split_ini'], 
                ['Transpose_0', 'Transpose_1', 'Transpose_2']
            )

# 'Add_9:0,1;Reshape_10'表示待new_add的输出同时作为Add_9的第0,1个输入以及Reshape_10的输入
new_add = g.add_node('new_add', 'Add')
g.connect_node(
                new_add,
                ['Conv_7', 'Conv_8'], 
                ['Add_9:0,1;Reshape_10']
            )
```

</details>

### 删除节点

**remove(name, maps={0:0}) -> bool**

- 删除图中指定节点，支持自动连边。
- `name(str)` - 待删除节点名称 \
  `maps(Dict[int])` - 可选参数。`maps` 中的`key` 值表示待删除节点的输入下标，`value` 值为待删除节点的输出下标。若不提供 `maps`，默认将前驱节点的第一个输出和后继节点的第一个输入相连；若 `maps={}`，仅删除指定节点，不连边。

<details>
  <summary> sample code </summary>

```python
g.remove('Cast_0') # 删除节点， 默认将第0个输入和第0个输出相连
g.remove('Split_0', {}) # 删除节点，不连边
g.remove('Node_text', {0:0,1:1}) # 删除节点，将节点的第0个输入和第0个输出相连，第1个输入和第1个输出相连
```

</details>

### 基础功能

**parse(path_or_bytes) -> BaseGraph**

- 加载模型，将模型解析为图。
- `path_or_bytes(Union[str, ModelProto, GraphProto])` - 输入可以是 onnx 模型文件路径，也可以是 onnx 框架中的 `ModelProto` 或 `GraphProto`。

**save(path)**

- 保存模型，保存前自动对节点进行拓扑排序。
- `path(str)` - 指定 onnx 文件的保存路径。

**update_map()**

- 手动连边后调用，用于更新前后节点关系。

**toposort()**

- 对算子节点进行拓扑排序，若图存在环路报错。

### 实用功能

**infershape()**

- 对 [onnx.shape_inference.infer_shapes](https://github.com/onnx/onnx/blob/main/onnx/shape_inference.py#L14) 的封装，用于形状推断。

**simplify(\*\*kwargs) -> BaseGraph**

- 对 [onnxsim.simplify](https://github.com/daquexian/onnx-simplifier/blob/master/onnxsim/onnx_simplifier.py#L91) 的封装，用于模型简化。

**extract(save_path, input_name_list, output_name_list) -> BaseGraph**

- 对 [onnx.utils.extract_model](https://github.com/onnx/onnx/blob/main/onnx/utils.py#L168) 的封装，用于模型截断。
- `save_path(str)` - 指定截取后模型 onnx 文件的保存路径。
- `input_name_list(List[str])` - 指定模型截取的输入边。
- `output_name_list(List[str])` - 指定模型截取的输出边。

<details>
  <summary> sample code </summary>

```python
# 形状推断并保存
g.infershape()
g.save('inferred_model.onnx')

# 模型简化并保存
g_sim = g.simplify()
g_sim.save('simplified_model.onnx')

# 模型截取并保存
g.extract('extracted_model.onnx', ['5'], ['12'])

```

</details>

### 属性

| API             | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| g.inputs        | 只读属性，返回整网输入节点组成的列表 `List[PlaceHolder]`     |
| g.outputs       | 只读属性，返回整网输出节点组成的列表 `List[PlaceHolder]`     |
| g.nodes         | 只读属性，返回整网算子节点组成的列表  `List[Node]`           |
| g.initializers  | 只读属性，返回整网常量节点组成的列表 `List[Initializer]`     |
| g.value_infos   | 只读属性，返回整网形状信息组成的列表  `List[PlaceHolder]`    |
| g.opset_imports | 可读写属性，返回算子版本信息 `List[OperatorSetIdProto]`<br>修改时输入版本号 `int`，比如 `g.opset_imports = 13` |

## BaseNode API

### 公共属性/方法

| API          | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| node.name    | 可读性属性，获取和修改节点名称                               |
| node.op_type | 只读属性，获取节点类型。<br>注：除算子节点外，输入/输出节点的类型名称为 `'PlaceHolder'`，常量节点的类型名称为`'Initializer'`。 |
| print(node)  | 打印节点信息                                                 |

### 常量节点

| API       | 说明                                                         |
| --------- | ------------------------------------------------------------ |
| ini.value | 可读写属性，借助 `numpy.ndarray` 获取和修改常量节点的具体数值 |

### 输入/输出节点

| API      | 说明                                                     |
| -------- | -------------------------------------------------------- |
| ph.dtype | 可读写属性，用 `numpy.dtype` 表示输入/输出节点的数据类型 |
| ph.shape | 可读写属性，用 `List[int]` 表示输入/输出节点的维度信息   |

### 算子节点

| API                           | 说明                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| op.inputs                     | 可读写属性，用 `List[str]` 表示算子节点输入列表              |
| op.outputs                    | 可读写属性，用 `List[str]` 表示算子节点输出列表              |
| op.attrs                      | 可读写属性，用 `Dict[str, object]` 表示算子节点属性          |
| op.get_input_id(input_name)   | 方法，`input_name` 为算子节点的某个输入名称，返回该输入对应的下标 |
| op.get_output_id(output_name) | 方法，`output_name` 为算子节点的某个输出名称，返回该输出对应的下标 |
