# chip.envs：仿真环境与观测编码

分两层：

| 层 | 文件 | 依赖 | 状态 |
|---|---|---|---|
| 几何编码（纯 CPU） | `geometry.py`、`conversions.py` | numpy、trimesh | 已写，29 个测试通过 |
| IsaacLab 环境 | `isaaclab/`（待写） | IsaacLab、GPU | 等集群环境 |

把几何编码和 IsaacLab 分开，是为了能在没有 GPU 的机器上开发和测试大部分逻辑。

## 几何编码（论文 3.4、附录 C）

### 局部高度图 `SceneHeightField`
- 每个场景**预计算一次**稠密全局高度场（光线自上而下投射，默认 5 cm 分辨率），存成 npz 缓存。
- 运行时 `local_heightmap(base_pos, yaw)` 在机器人周围 16×16、1.6 m × 1.6 m 的网格上双线性采样，
  减去机器人根高度，得到"相对机器人"的高度。
- 网格第 0 维是机器人前方，第 1 维是左方。测试里验证了转身后桌子位置随之翻转。

### 物体 BPS `ObjectBPS`
- `make_basis_points(512)`：固定种子，在单位球内撒 512 个基点，全项目共用一组。
- 每个物体预采样表面点一次；运行时把基点按"机器人位置 + 朝向"摆放，再变换到物体系求最近距离。
- 测试验证了物体和机器人一起做刚体变换时编码不变。

### 身体-物体几何 `body_to_object_vectors`
- 每个 body 到物体表面最近点的向量，(B, 3)。
- **假设**：论文的 96 维 = 32 body × 3。具体用哪 32 个 body 待定，写在 Stage III 配置里。

### 格式转换 `conversions.py`
- 四元数 (x,y,z,w) ↔ (w,x,y,z)。GMR / scipy 用前者，IsaacLab 用后者。
- `JointReorder`：按关节名在 GMR 顺序和 IsaacLab 顺序之间重排，运行时从 env 读名字，不硬编码。

## IsaacLab 层（待写）

需要做的：
1. G1 29-DoF 的 ArticulationCfg（IsaacLab 自带 `unitree.py` 里有 G1，确认是 29 还是 23 DoF 版本）。
2. 场景加载：TRUMANS 的 OBJ 网格转 USD，作为静态碰撞体。注意 Y-up → Z-up（`load_mesh(yup_to_zup=True)`）。
3. 物体：OMOMO 的 OBJ 转 USD，作为刚体，质量和摩擦系数要定（论文没写）。
4. 观测拼装：本体感知 + 上面三种几何编码 + 目标。
5. 参考动作回放（Stage I 的 RSI 和跟踪奖励要用）。
6. 200 Hz 物理 / 50 Hz 控制，PD 增益（论文没写，先用 IsaacLab G1 默认值）。

第一个验收点：把一条 `RobotMotion` 在 IsaacLab 里**纯运动学回放**（直接设关节角，不开物理），
肉眼看动作对不对、脚是否踩在地上、场景是否朝向正确。这一步不需要训练，环境搭好就能做。

## 2026-11-21 回放验证结果（IsaacLab 2.3.0, G1_29DOF_CFG）

- 资产 43 个关节 = 我们的 29 个 + 14 个 Dex3 手指；关节顺序按树交错（左右腿、腰、手臂混排），
  必须按名字映射：`JointReorder(G1_DOF_NAMES, robot.joint_names, fill_missing=0.0)`。
- 根部四元数直接用 GMR 的 (x,y,z,w)→(w,x,y,z) 即可，**不要**加 90° 补偿。
- 同一片段的脚高：MuJoCo 0.014/0.009 m vs IsaacLab 0.021/0.025 m，一致。
- RunPod 容器没有图形能力，`--enable_cameras` 会段错误；无头物理仿真正常。
- `SimulationApp.close()` 在无头模式偶尔挂死，脚本里加了 20 s 看门狗强制退出。
