# 知识库列表

---

## 合并连续Concat算子(KnowledgeMergeConsecutiveConcat)

### 原理

有些onnx计算图内存在一些连续Concat算子，由于Concat算子可以接受任意个输入，当这些Concat算子合并的轴为同一个时，可以将这些连续的Concat算子合并成一个，以加快推理速度。

### 示意图

```mermaid
graph TD
    subgraph After
        X0[X] --> C(combined_concat)
        Y0[Y] --> C
        Z0[Z] --> C
    end

    subgraph Before
        X1[X] --> C0(concat0)
        Y1[Y] --> C0
        Z1[Z] --> C1(concat1)
        C0 --> C1
    end
```

如图所示，concat0和concat1被合并为combined_concat，concat0和concat1的输入都作为combined_concat的输入。

---

## 合并连续Slice算子(KnowledgeMergeConsecutiveSlice)

### 原理

有些onnx计算图内存在一些连续Slice算子，由于Slice算子可以接受向量作为切片参数，当这些Slice算子切分的轴各不相同时，可以将这些连续的Slice算子合并成一个，以加快推理速度。

### 示意图

```mermaid
graph TD
    subgraph After
        P1[PreNode] --> S(combined_slice)
        S --> N1[PostNode]
    end

    subgraph Before
        P0[PreNode] --> S0(slice0)
        S0 --> S1(slice1)
        S1 --> N0[PostNode]
    end
```

如图所示，slice0和slice1被合并为combined_slice，他们的切片参数被合并为combined_slice的切片参数。

---

## 拆分QKV结构内MatMul算子(KnowledgeSplitQKVMatmul)

### 原理

在transformer等模型中，存在很多固定的MatMul+Reshape+Transpose+Gather组合，在满足一定的前提条件下，通过矩阵的分块乘法，将QKV结构内的矩阵乘法均分为若干条分支，可以提升计算图的并行度以及消除部分Transpose算子，达到加快推理速度的目的。

### 示意图

如下左边是修改之前，右边是修改之后的计算图。

```mermaid
graph TD
    subgraph After
        A[PreNode] --> B0(MatMul0)
        A[PreNode] --> B1(MatMul1)
        A[PreNode] .-> B2(MatMul...)
        B0 --> C0(ElementWise0)
        B1 --> C1(ElementWise1)
        B2 .-> C2(ElementWise...)
        C0 .-> |zero/more| C0
        C1 .-> |zero/more| C1
        C2 .-> |zero/more| C2
        C0 --> D0(Reshape0)
        C1 --> D1(Reshape1)
        C2 .-> D2(Reshape...)
        D0 --> E0(Transpose0_0 + Transpose1)
        D1 --> E1(Transpose0_1)
        D2 .-> E2(Transpose...)
        E0 --> H0[PostNode0]
        E1 --> H1[PostNode1]
        E2 .-> H2[PostNode...]
    end

    subgraph Before
        M[PreNode] --> N(MatMul)
        N --> O(ElementWise)
        O .-> |zero/more| O
        O --> P(Reshape)
        P --> Q0(Transpose)
        Q0 --> R0(Gather0)
        Q0 --> R1(Gather1)
        Q0 .-> R2(Gather...)
        R0 --> Q1(Transpose1)
        R1 --> S1[PostNode1]
        R2 .-> S2(...)
        Q1 --> S0[PostNode0]
        S2 .-> Z2[PostNode...]
    end
```

如图所示，左边是常见于transformer模型内的一种结构，矩阵乘法的结果通过Reshape/Transpose算子组合之后，被数个Gather算子平分。修改之后，MatMul算子被直接切分为若干个MatMul算子，不再需要Gather算子来做切分。

相对于特定的模型内的结构，这里对pattern主要做了两个泛化：

1. ElementWise算子可以是Add/Sub/Mul/Div四种，数量为M，M>=0
2. Gather算子数量为N，N>=2

根据矩阵乘法，为了能够拆分，子图还需要满足以下条件：

1. Reshape/Transpose/Gather的组合需要平分矩阵乘法的结果。
2. MatMul和Reshape算子的第二个输入必须是常数即Initializer，否则无法进行判断。
3. 子图中除了PreNode和PostNode节点外，均不能有额外的输出节点

示意图内的分支有两种，他们的区别是Gather算子后是否为transpose算子，在拆分后，Gather算子被消除，因此如果Gather算子前后均为Transpose算子，则这两个Transpose算子可以合并成一个，否则不需要特殊处理。

---

## 交换大shape卷积算子的H/W轴(KnowledgeTransposeLargeInputConv)

### 原理

部分音频模型中存在输入形状非常大的卷积算子，严重影响推理性能，利用NPU卷积操作中H轴相比W轴有更好的tiling策略的特点，我们交换卷积算子的H轴和W轴，从而大幅提升推理速度。目前该知识库只适用于下面特定的子图结构。

### 示意图

```mermaid
graph TD
    subgraph After
        P[PreNode] --> T0(transpose_pre)
        T0 --> S0(selu0)
        S0 --> C0(LargeInputConv0_transposed)
        C0 --> A(add0)
        S0 --> C1(LargeInputConv1_transposed)
        C1 --> S1(selu1)
        S1 --> C2(LargeInputConv2_transposed)
        C2 --> A(add0)
        A --> T1(transpose_post)
        T1 --> N[PostNode]
    end

    subgraph Before
        M[PreNode] --> K0(selu0)
        K0 --> X0(LargeInputConv0)
        X0 --> Z(add0)
        K0 --> X1(LargeInputConv1)
        X1 --> K1(selu1)
        K1 --> X2(LargeInputConv2)
        X2 --> Z
        Z --> O[PostNode]
    end
```

如图所示，我们在子图前后均加上Transpose算子，将子图中间的卷积算子均进行H/W轴转置，由于Selu算子不会改变输入输出的shape，所以不需要处理。

---

## 拆分大kernel卷积算子(KnowledgeSplitLargeKernelConv)

### 原理

部分音频和OCR模型中，存在卷积核特别大的卷积算子，严重影响推理速度。通过将该卷积算子通过Slice+Conv+Unsqueeze+Concat+ReduceSum组合拆分成多个卷积算子，可以有效提升推理速度。

### 示意图

```mermaid
graph TD
    subgraph After
        P[PreNode] --> S0(slice0)
        P --> S1(slice1)
        P .-> S2(slice...)
        S0 --> C0(conv0)
        S1 --> C1(conv1)
        S2 .-> C2(conv...)
        C0 --> U0(unsqueeze0)
        C1 --> U1(unsqueeze1)
        C2 .-> U2(unsqueeze...)
        U0 --> K(concat)
        U1 --> K
        U2 .-> K
        K --> R(reduce_sum)
        R --> H[PostNode]
    end

    subgraph Before
        Z[PreNode] --> X(LargeKernelConv)
        X --> Y[PostNode]
    end
```

如图所示，我们将大kernel卷积算子根据kernel拆分为若干个，根据其kernel切片的移动范围将输入进行Slice切片，再将各个拆分后的卷积算子结果全部加起来，即得到了等效的结果。

---

## TopK算子输入输出类型修复(KnowledgeTopkFix)

### 原理

在ONNX标准中，TopK算子的输入K和输出indices其类型定义为int64[]，而在om的实现中为int32[]，由于ATC转换工具在某些情况下未正确处理这个问题，因此如果模型中存在TopK算子且其k或indices存在类型不匹配，则ATC有可能报错退出，无法转换该模型。这里通过对TopK算子相应的输入输出进行类型转换来处理这个问题。经过修改后，模型不再符合ONNX标准，因此仅推荐模型转换om时遇到TopK类型问题时才启用这个知识库。

### 示意图

```mermaid
graph TD
    subgraph After
        X(Node0) --> Y(Cast)
        Y --> |k| D(TopK)
        D --> E(Node1)
        D --> |indices| F(Cast)
        F --> G(Node2)
    end

    subgraph Before
        Z(Node0) -->|k| A(TopK)
        A --> B(Node1)
        A --> |indices| C(Node2)
    end
```

---

## 空slice修复(KnowledgeEmptySliceFix)

### 原理

ONNX标准中，Concat算子支持输入空张量，但是om的Concat算子实现某些情况下不支持，此时ATC转换会失败。一种已知的情况是输出空张量的slice算子和concat算子组合，本知识库通过删除该Slice算子，并根据前后的连接情况不同而做不同的处理，来规避这个问题。

### 示意图

```mermaid
graph TD
    subgraph After 

        F(PreNode) --> G(NormalSlice)
        G --> H(NextNode)
    end

    subgraph Before
        A(PreNode) --> B(EmptySlice)
        A --> C(NormalSlice)
        B --> D(Concat)
        C --> D
        D --> E(NextNode)

    end
```
---

## Resize算子mode使用最近邻(KnowledgeResizeModeToNearest)

### 原理

部分推理模型中使用了双线性插值法做resize，经分析导致精度回归异常，使得各别图片存在误差。此类优化可扩展至linear->nearest、cubic->nearest、area->nearest等自定义转换场景。
### 可支持场景
Resize算子mode类型为linear、cubic、area的场景。 
### 示意图

```mermaid
graph TD
    subgraph After
        X(Node0) --> D(Resize mode:nearest)
        D --> E(Node1)
    end

    subgraph Before
        Z(Node0) -->A(Resize mode:linear)
        A --> B(Node1)
    end
```
---

## Split算子替换Gather算子(KnowledgeGatherToSplit)

### 原理

部分推理模型中使用了多个Gather算子对同一个数据进行切分，经分析Gather算子indices连续的情况下，例如该场景：y1=x[:3]，y2=x[3:6]，y3=x[6:9]，可使用一个Split算子进行替换。
### 可支持场景
各Gather算子axis相同，indices为0开始的连续一维向量且切分数据不相交的场景。例如：三个Gather算子indices分别为[0]、[1]、[2]；三个Gather算子indices分别为[0, 1]、[2, 3]、[4, 5]。
### 示意图

```mermaid
graph TD
    subgraph After
        X(Node0) --> D(Split)
        D --> E(Node1)
        D --> F(Node2)
        D --> G(Node3)
    end

    subgraph Before
        Z(Node0) --> A(Gather0)
        Z --> B(Gather1)
        Z --> C(Gather2)
        A --> H(Node1)
        B --> I(Node1)
        C --> J(Node2)
    end
```

---

## 动态shape模型Reshape算子优化(KnowledgeDynamicReshape)

### 原理

由于动态shape模型的输入shape不固定，onnx模型中大部分Reshape算子的shape值需要在执行过程中计算得出。这就会导致两个问题：1、计算Reshape的shape值引入了很多小算子，增加了调度耗时；2、shape值未知，导致Reshape算子的infershape依赖前置算子的输出（Reshape的infershape需要等前置算子执行完才能开始），会打断调度流水。如果能够计算出Reshape的shape值，就可以减少很多非必须的算子，从而有效提高模型性能。Reshape算子有两个特性可以帮助我们实现：如果shape某一个轴大小没有变化，可以设置为0；如果输入只有一个动态轴，该轴可以赋值为-1。然后通过静态推导方法，对模型指定不同的固定输入，再根据Reshape算子推导出的输入和输出计算出该算子的shape值。

### 参数配置

因为动态shape模型输入存在差异，不同模型输入可接受的范围不同，需要用户手动配置动态轴的输入范围。可以打开配置文件auto_optimizer/model.cfg，修改input_shape_range值，动态轴的取值范围通过“~”符号的方式连接，如果动态轴是固定值的倍数，则需要在固定值后面加上“*”符号，详细的描述请移步model.cfg的配置说明。

### 示意图

```mermaid
graph TD
    subgraph After
        Y(Input)
        N(Node0)
        Re(Reshape)

        Y --> N
        N --> Re
    end

    subgraph Before
        X(Input)
        N0(Node0)
        N1(Node1)
        N2(Node2)
        R(Reshape)

        X --> N0
        X --> N1
        N0 --> R
        N1 --> N2
        N2 --> R
    end
```

---

## AveragePool算子大kernel size和stride拆分(KnowledgeAvgPoolSplit)

### 原理

因为Ascend指令集的限制，AvgPool算子的kernel size最大不能超过255，如果超过会额外插入Transdata、Transpose等算子进行shape转换，导致性能下降。可以优化onnx模型，将AveragePool算子拆分成等价的多个串联的AveragePool算子，比如AveragePool("kernel_shape": [32, 64], "stride": [32, 64])，可以拆分成AveragePool_0("kernel_shape": [8, 16], "stride": [8, 16])和AveragePool_1("kernel_shape": [4, 4], "stride": [4, 4])，拆分后转成om模型，可以消除多余的Transdata、Transpose等算子。

### 示意图

```mermaid
graph TD
    subgraph After
        X(Input)
        A0(AveragePool_0)
        A1(AveragePool_1)
        A2(AveragePool_2)

        X --> A0
        A0 --> A1
        A1 --> A2
    end

    subgraph Before
        Y(Input)
        A(AveragePool)

        Y --> A
    end
```

## Conv1d 转 Conv2d (KnowledgeConv1d2Conv2d)

### 原理

在 onnx 图转换为 om 图过程中，Conv1d 算子会转换为 Conv2d 算子，并在算子前后插入 Transdata 进行数据类型转换，图中的多个 Conv 算子就会导致多次数据类型转换引起性能损失。本知识库识别多个连通的 (Conv1d | Element-wise) 类型的算子为一个子图，并在子图的所有输入前插入 Unsqueeze 算子，所有输出后插入 Squeeze 算子将子图进行整体升维从而减少数据类型转换。

### 示意图

```mermaid
graph TD
    subgraph After
        subgraph " 1+"
            C2("Conv(2d)") --> E2(Element-wise)
            E2 .->|0+| E2
        end
        P2[PreNode] --> U(Unsqueeze)
        U --> C2
        E2 --> S(Squeeze)
        S --> O2[PostNode]
    end

    subgraph Before
        subgraph 1+
            C1("Conv(1d)") --> E1(Element-wise)
            E1 .->|0+| E1
        end
        P1[PreNode] --> C1
        E1 --> O1[PostNode]
    end
```

如图所示，Before 对应的是优化之前的 onnx 图，图中的 `Conv(1d) --> Element-wise` 结构代表由若干个 Conv 算子和 Element-wise 类型算子连通形成的子图。整个优化过程就是在子图的所有输入前插入 Unsqueeze 算子，所有输出后插入 Squeeze 算子将子图进行整体升维。

```mermaid
graph TD
    subgraph After
        subgraph " Subgraph"
            C4("Conv(2d)") --> L2(LeakyRelu)
            L2 --> C5("Conv(2d)")
            C6("Conv(2d)") --> A2(Add)
        end
        P2[PreNode] --> U1(Unsqueeze)
        U1 --> C4
        P2 --> U2(Unsqueeze)
        U2 --> C6
        C5 --> A2
        A2 --> S(Squeeze)
    end

    subgraph Before
        subgraph Subgraph
            C1("Conv(1d)") --> L1(LeakyRelu)
            L1 --> C2("Conv(1d)")
            C3("Conv(1d)") --> A1(Add)
        end
        P1[PreNode] --> C1
        P1 --> C3
        C2 --> A1
    end
```

上图是一个更具体的示例，可以看到 Subgraph 中就是符合要求的子图，优化后子图的输入输出插入了 Unsqueeze 和 Squeeze 算子。

## 数据类型转换 (KnowledgeTypeCast)

### 原理

一些模型中使用了 int64 等较高精度的数据类型，实际模型并不需要这么高的精度进行推理，因此可以通过特定的类型转换策略对 onnx 图中的数据类型进行转换，从而提升图推理的性能。基本原理为找到 onnx 图中所有满足类型要求并可泛型的子图，通过在子图前后插入 Cast 算子将子图的数据类型转换为目标类型。

### 已支持场景

#### 已支持类型转换策略

后续可扩展为更复杂的转换策略

- int64 -> int32
- float64 -> float32

#### 已支持类型转换的算子

完全泛型算子：

- Mul
- Add
- Sub
- Div
- Abs
- Tanh
- LeakyRelu
- Relu
- Sigmoid
- BatchNormalization
- ReduceSum
- Concat
- Gemm
- Split
- Slice
- Transpose

半泛型算子：

- Expand: GenericIO([0], [0])
- Less: GenericIO([0, 1], [])
- Gather: GenericIO([0], [0])
- Shape: GenericIO([0], [])
- Where: GenericIO([1, 2], [0])
- Equal: GenericIO([0, 1], [])
- Reshape: GenericIO([0], [0])
- Tile: GenericIO([0], [0])
- ScatterND: GenericIO([0, 2], [0])
- Unsqueeze: GenericIO([0], [0])
- Squeeze: GenericIO([0], [0])

### 示意图

```mermaid
graph TD
    subgraph After
        subgraph "Subgraph"
            V2(VaradicNode) .->|1+| V2
        end
        P2[PreNode] --> C3("Cast {to: dst_type}")
        C3 --> V2
        V2 --> C4("Cast {to: ori_type}")
        C4 --> O2[PostNode]
    end

    subgraph Before
        subgraph "Subgraph "
            V1(VaradicNode) .->|1+| V1
        end
        P1[PreNode] --> V1
        V1 --> O1[PostNode]
    end
```

如上图所示，原图中存在满足匹配条件的子图，知识库通过在子图的所有输入前插入 `Cast {to: dst_type}` 算子，在输出后插入 `Cast {to: ori_type}` 算子，将子图内部的可泛型算子转换为目标类型。

```mermaid
graph TD
    subgraph After
        subgraph "Subgraph "
            S3(Shape) -->|int64| E2(Expand)
            E2 -->|S| A2(Add)
            A2 -->|S| S4(Shape)
            C2(("Const(S)")) -->|S| A2
        end
        P2[PreNode] -->|T| Cast1("Cast {to: s}") -->|S| S3
        P2 -->|T| Cast2("Cast {to: s}")  -->|S| E2
        S4 -->|int64| O2(PostNode)
    end

    subgraph Before
        subgraph "Subgraph"
            S1(Shape) -->|int64| E1(Expand)
            E1 -->|T| A1(Add)
            A1 -->|T| S2(Shape)
            C1(("Const(T)")) -->|T| A1
        end
        P1[PreNode] -->|T| S1
        P1 -->|T| E1
        S2 -->|int64| O1(PostNode)
    end
```

上图是一个更具体的示例，有几个细节需要注意：

1. 子图作为一个整体，只需要在子图的外部输入前插入 `Cast {to: s}` 算子，子图内部的可泛型算子也会转换为目标类型
2. Add 算子的常量输入节点内部的数据也会转换为目标类型
3. Shape 算子的输出不能泛型，因此 Shape 算子的输出作为图输出也不会插入 `Cast {to: T}` 算子

## Cast 算子合并 (KnowledgeMergeCasts)

### 原理

本知识库是对 KnowledgeTypeCast 知识库的一个补充优化，目的是合并图结构中多余的 Cast 算子，从而减少类型转换操作，也可以方便后续其他知识库进行结构合并。

### 示意图

Cast 算子合并可以归纳为以下三种方法：

1. 同属性的兄弟 Cast 算子合并

```mermaid
graph TD
    subgraph After
        A2(Add) --> C3("Cast{to: T}") --> M2(Mul)
        C3 --> S2(Sub)
    end

    subgraph Before
        A1(Add) --> C1("Cast{to: T}") --> M1(Mul)
        A1 --> C2("Cast{to: T}") --> S1(Sub)
    end
```

2. 单分支路径上的父子 Cast 算子合并

```mermaid
graph TD
    subgraph After
        A2(Add) --> C3("Cast{to: S}")
    end

    subgraph Before
        A1(Add) --> C1("Cast{to: T}") --> C2("Cast{to: S}")
    end
```

3. 根节点后的 Cast 算子如果与输出类型相同可以消除

```mermaid
graph TD
    subgraph After
        A2(Add) -->|T| M2(Mul)
    end

    subgraph Before
        A1(Add) -->|T| C1("Cast{to: T}") --> M1(Mul)
    end
```

结合以上三种方法，对 Cast 节点树进行递归处理就可以合并多余的 Cast 节点


## BatchNormalization折叠 (KnowledgeBNFolding)

### 原理及示意图

BatchNormalization的数学表示如下：

$$
\mathbf{Y} = \frac{\mathbf{X} - \textrm{E}[\mathbf{X}]}{\sqrt{\textrm{Var}[\mathbf{X}] + \epsilon}} \times \gamma + \beta
$$

其中 $\mathbf{X}$ 为需要normalize的数据，shape为 $[N, C, D_1, D_2, ..., D_n]$ ， $\mathbf{Y}$ 为算子输出， $\textrm{E}[\mathbf{X}]$ 为输入均值mean， $\textrm{Var}[\mathbf{X}]$ 为输入方差var，$\gamma$ 为输入scale，$\beta$ 为输入bias，$\textrm{E}[\mathbf{X}], \textrm{Var}[\mathbf{X}], \gamma, \beta$ 的shape均为 $[C]$

其中的 $\epsilon$ 为BN的一个属性，是一个float常量

当前后有Transpose且两个Transpose可以互相抵消时，整个TR/BN/TR结构的数学表示可以写成：

$$
\mathbf{Y} = \left[\frac{\mathbf{X}^\mathsf{T} - \textrm{E}[\mathbf{X}]}{\sqrt{\textrm{Var}[\mathbf{X}] + \epsilon}} \times \mathbf{\gamma} + \beta \right]^\mathsf{T}
$$

化简为

$$
\mathbf{Y} = \mathbf{W} \times \mathbf{X} + \mathbf{B}
$$

其中

$$ \mathbf{W} = \left[\frac{\gamma}{\sqrt{\textrm{Var}[\mathbf{X}] + \epsilon}} \right]^\mathsf{T} $$
$$ \mathbf{B} = \left[\beta - \frac{\gamma \times \textrm{E}[\mathbf{X}]}{\sqrt{\textrm{Var}[\mathbf{X}] + \epsilon}} \right]^\mathsf{T} $$

因此当这里的输入 $\textrm{E}[\mathbf{X}], \textrm{Var}[\mathbf{X}], \gamma, \beta$ 都为常量时，可以进行常量折叠，将这里的TR/BN/TR组合替换为Mul/Add组合, $\mathbf{W}$ 和 $\mathbf{B}$ 分别为Mul和Add算子的输入。如图所示

```mermaid
graph TD
    subgraph After
        F[input] --> G(Mul)
        subgraph "Subgraph "
                G --> H(Add)
        end
        H --> I(output)
    end

    subgraph Before
        A[input] --> B(Transpose0)
        subgraph "Subgraph"
                B --> C(BatchNormalization)
                C --> D(Transpose1)
        end
        D --> E(output)
    end
```

当没有Transpose时，FusionPass会进行一些类似Conv+BatchNormalization的融合，原理上是一样的，尝试了一些模型，部分性能有劣化，故不考虑。
