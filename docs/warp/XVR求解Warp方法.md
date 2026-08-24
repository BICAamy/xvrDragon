# XVR 求解 Warp 方法

## 1. 目的

在使用 XVR 的 WBCT foundation checkpoint 对新的脊柱 CT 做 patient-specific finetune 或后续大规模微调时，需要把每个新 CT 的坐标系映射到 foundation model 的参考坐标系。

XVR 的 `--warp` 定义是：

> 将输入 CT 映射到 checkpoint 所使用的 reference frame。

当前作者没有公开 WBCT foundation 所使用的原始 whole-body template CT，因此不能直接做：

```text
新 CT -> WBCT template
```

目前采用一个已经验证可行的“桥接”方案：

```text
新 CT
  ↓
DeepFluoro subject01
  ↓
WBCT foundation reference frame
```

我们已经具备桥接所需的两个固定文件：

```text
experiments/data/template_bridge/deepfluoro_subject01/
├── volume.nii.gz
└── warp.mat
```

其中：

- `volume.nii.gz`：DeepFluoro subject01 CT；
- `warp.mat`：作者提供的 DeepFluoro subject01 → WBCT foundation reference transform。

后续所有新 CT 都可以复用这两个固定文件。

---

## 2. 坐标关系

作者提供的变换是：

$$
T_{\text{WBCT foundation reference}\leftarrow\text{DeepFluoro subject01}}
$$

也就是：

```text
DeepFluoro subject01 -> foundation reference
```

对应文件：

```text
experiments/data/template_bridge/deepfluoro_subject01/warp.mat
```

接下来对每个新的脊柱 CT，使用 ANTs 做刚体配准：

得到：

$$
T_{新CT\leftarrow DeepFluoro}
$$


也就是：

```text
DeepFluoro subject01 -> 新 CT
```

但我们最终需要的是：

$$
T_{\text{WBCT foundation reference}\leftarrow\text{新CT}}
$$
因此先求逆：

$$
T_{\text{DeepFluoro subject01}\leftarrow\text{新CT}}
=
T_{\text{新CT}\leftarrow\text{DeepFluoro subject01}}^{-1}
$$
最终：

$$
\boxed{
T_{\text{WBCT foundation reference}\leftarrow\text{新CT}}
=
T_{\text{WBCT foundation reference}\leftarrow\text{DeepFluoro subject01}}
T_{\text{DeepFluoro subject01}\leftarrow\text{新CT}}
}
$$
即：

$$
\boxed{
T_{\text{WBCT foundation reference}\leftarrow\text{新CT}}
=
T_{\text{WBCT foundation reference}\leftarrow\text{DeepFluoro subject01}}
T_{\text{新CT}\leftarrow\text{DeepFluoro subject01}}^{-1}
}
$$
这就是每个新 CT 所需要的 `warp.mat`。

---

## 3. 为什么不能直接把 ANTs 参数当成普通 4×4

ANTs / ITK affine transform 的形式是：

$$
y=A(x-c)+c+t
$$
其中：

- \(M_{3\times3}\)：ANTs / ITK transform 中的 3×3 线性部分
- \(c\)：`FixedParameters` 中的变换中心
- \(t_{\text{ITK}}\)：`Parameters` 中最后三个平移参数

展开为：

$$
y=Ax+(-Ac+c+t)
$$
因此真正写入普通齐次矩阵的平移项应为：

$$
t_{\text{global}}=-Ac+c+t
$$
对应：

$$
T=
\begin{bmatrix}
M_{3\times3}&t_{\text{global}}\\
0&1
\end{bmatrix}
$$
不要直接把 `transform.parameters[9:12]` 当作普通 4×4 的平移列。

XVR 自己的 `src/xvr/utils/ants.py -> get_4x4()` 也是按照这个规则解析 transform。

---

## 4. 单个 CT 的完整求解流程

假设当前病例目录：

```text
data/vertebra/1001/
└── volume.nii.gz
```

### Step 1：求 DeepFluoro subject01 → 新CT 的刚体变换

```bash
python - <<'PY'
from xvr.utils import ants_rigid_register

fixed = "data/vertebra/1001/volume.nii.gz"
moving = "experiments/data/template_bridge/deepfluoro_subject01/volume.nii.gz"
output = "data/vertebra/1001/deepfluoro_to_newct.mat"

ants_rigid_register(
    fix_filename=fixed,
    mov_filename=moving,
    savepath=output,
)

print("WROTE:", output)
PY
```

得到：

```text
data/vertebra/1001/deepfluoro_to_newct.mat
```

其数学含义：

$$
T_{\text{新CT}\leftarrow\text{DeepFluoro subject01}}
$$

---

### Step 2：组合得到 新CT → WBCT foundation reference 的 warp

```bash
python - <<'PY'
import ants
import numpy as np

DEEPFLUORO_TO_WBCT_WARP = (
    "experiments/data/template_bridge/"
    "deepfluoro_subject01/warp.mat"
)

DEEPFLUORO_TO_NEW_CT_WARP = "data/vertebra/1001/deepfluoro_to_newct.mat"
OUTPUT = "data/vertebra/1001/warp.mat"


def ants_to_matrix(path):
    tfm = ants.read_transform(path)

    p = np.asarray(tfm.parameters, dtype=np.float64)
    c = np.asarray(tfm.fixed_parameters, dtype=np.float64)

    affine_3x3 = p[:9].reshape(3, 3)
    itk_translation = p[9:12]

    global_translation = (
        -affine_3x3 @ c + c + itk_translation
    )

    transform_4x4 = np.eye(4, dtype=np.float64)
    transform_4x4[:3, :3] = affine_3x3
    transform_4x4[:3, 3] = global_translation

    return transform_4x4


# DeepFluoro subject01 -> WBCT foundation reference
T_WBCT_reference_from_DeepFluoro = ants_to_matrix(DEEPFLUORO_TO_WBCT_WARP)

# DeepFluoro subject01 -> 新CT
T_newCT_from_DeepFluoro = ants_to_matrix(DEEPFLUORO_TO_NEW_CT_WARP)

# 新CT -> DeepFluoro subject01
T_DeepFluoro_from_newCT = np.linalg.inv(T_newCT_from_DeepFluoro)

# 新CT -> WBCT foundation reference
T_WBCT_reference_from_newCT = (
    T_WBCT_reference_from_DeepFluoro @ T_DeepFluoro_from_newCT
)

affine_3x3 = T_WBCT_reference_from_newCT[:3, :3]
global_translation = T_WBCT_reference_from_newCT[:3, 3]

parameters = np.concatenate([
    affine_3x3.reshape(-1),
    global_translation,
])

out_tfm = ants.create_ants_transform(
    transform_type="AffineTransform",
    precision="float",
    dimension=3,
    parameters=parameters,
    fixed_parameters=np.zeros(3, dtype=np.float64),
)

ants.write_transform(out_tfm, OUTPUT)

print("WROTE:", OUTPUT)
print(T_WBCT_reference_from_newCT)
PY
```

得到：

```text
data/vertebra/1001/warp.mat
```

这个文件即可作为：

```bash
xvr train ... -w data/vertebra/1001/warp.mat
```

的输入。

---

## 5. 最终 warp 验证

生成 `warp.mat` 后，必须让 XVR 自己再读取一次：

```bash
python - <<'PY'
import numpy as np
from xvr.utils import get_4x4

warp = "data/vertebra/1001/warp.mat"
ct = "data/vertebra/1001/volume.nii.gz"

T = get_4x4(warp, ct, invert=False)

M = T.matrix.squeeze().detach().cpu().numpy()
R = M[:3, :3]
t = M[:3, 3]

print("XVR reframe =")
print(M)

print("\ndet(R) =", np.linalg.det(R))
print("\nR^T R =")
print(R.T @ R)

print("\nt =", t)
PY
```

XVR 的 `get_4x4()` 最后会把 affine 的 3×3 部分投影到最近的 \(SO(3)\)，因此最终应满足：

$$
\det(R)\approx1
$$
以及：

$$
R^TR\approx I
$$
如果这两项明显不成立，则不要进入训练。

注意：

```text
det(R) ≈ 1
R^T R ≈ I
```

只能说明最终变换是合法刚体，不能证明 CT 间的实际配准是正确的，因此还需要做影像 QC。

---

## 6. 配准 QC

推荐至少对部分病例做可视化检查。

将 DeepFluoro subject01 CT 变换到新CT网格：

```bash
python - <<'PY'
import ants

fixed_path = "data/vertebra/1001/volume.nii.gz"
moving_path = "experiments/data/template_bridge/deepfluoro_subject01/volume.nii.gz"
transform_path = "data/vertebra/1001/deepfluoro_to_newct.mat"

output_path = "data/vertebra/1001/deepfluoro_in_patient.nii.gz"

fixed = ants.image_read(fixed_path)
moving = ants.image_read(moving_path)

warped = ants.apply_transforms(
    fixed=fixed,
    moving=moving,
    transformlist=[transform_path],
    interpolator="linear",
)

ants.image_write(warped, output_path)

print("WROTE:", output_path)
PY
```

检查重点：

- 脊柱整体方向是否一致；
- 腰椎/目标椎体是否大致重合；
- 是否出现明显 90° / 180° 翻转；
- 是否有几百毫米级明显错位；
- 不同患者的椎体轮廓不可能完全一致，这是正常的。

这里的 bridge registration 目的不是获得最终临床级配准，而是建立 foundation model 所需的参考坐标关系。

---

# 7. 批量求解 warp

推荐的数据组织：

```text
data/spine_cases/
├── case0001/
│   └── volume.nii.gz
├── case0002/
│   └── volume.nii.gz
├── case0003/
│   └── volume.nii.gz
└── ...
```

固定使用 DeepFluoro subject01 作为桥接 CT：

```text
experiments/data/template_bridge/deepfluoro_subject01/
├── volume.nii.gz
└── warp.mat
```

下面的脚本会对所有病例依次生成：

```text
caseXXXX/
├── volume.nii.gz
├── deepfluoro_to_newct.mat
└── warp.mat
```

同时写出 `warp_batch_report.csv`。

建议保存为：

```text
scripts/build_wbct_warps.py
```

代码：

```python
from pathlib import Path
import csv

import ants
import numpy as np

from xvr.utils import ants_rigid_register, get_4x4


DEEPFLUORO_CT = Path(
    "experiments/data/template_bridge/"
    "deepfluoro_subject01/volume.nii.gz"
)

DEEPFLUORO_TO_WBCT_WARP = Path(
    "experiments/data/template_bridge/"
    "deepfluoro_subject01/warp.mat"
)

DATA_ROOT = Path("data/spine_cases")
REPORT = DATA_ROOT / "warp_batch_report.csv"


def ants_to_matrix(path):
    tfm = ants.read_transform(str(path))

    p = np.asarray(tfm.parameters, dtype=np.float64)
    c = np.asarray(tfm.fixed_parameters, dtype=np.float64)

    affine_3x3 = p[:9].reshape(3, 3)
    itk_translation = p[9:12]

    global_translation = (
        -affine_3x3 @ c + c + itk_translation
    )

    transform_4x4 = np.eye(4, dtype=np.float64)
    transform_4x4[:3, :3] = affine_3x3
    transform_4x4[:3, 3] = global_translation

    return transform_4x4


def write_affine_matrix(M, output):
    affine_3x3 = M[:3, :3]
    global_translation = M[:3, 3]

    parameters = np.concatenate([
        affine_3x3.reshape(-1),
        global_translation,
    ])

    tfm = ants.create_ants_transform(
        transform_type="AffineTransform",
        precision="float",
        dimension=3,
        parameters=parameters,
        fixed_parameters=np.zeros(3, dtype=np.float64),
    )

    ants.write_transform(tfm, str(output))


T_WBCT_reference_from_DeepFluoro = ants_to_matrix(DEEPFLUORO_TO_WBCT_WARP)

rows = []

cases = sorted(DATA_ROOT.glob("*/volume.nii.gz"))

print("Number of CTs:", len(cases))

for idx, new_ct in enumerate(cases, start=1):
    new_ct_dir = new_ct.parent
    case_id = new_ct_dir.name

    deepfluoro_to_newct_warp = new_ct_dir / "deepfluoro_to_newct.mat"
    warp = new_ct_dir / "warp.mat"

    print(f"\n[{idx}/{len(cases)}] {case_id}")

    try:
        # --------------------------------------------------
        # 1. DeepFluoro subject01 -> 新CT
        # --------------------------------------------------
        ants_rigid_register(
            fix_filename=str(new_ct),
            mov_filename=str(DEEPFLUORO_CT),
            savepath=str(deepfluoro_to_newct_warp),
        )

        T_newCT_from_DeepFluoro = ants_to_matrix(deepfluoro_to_newct_warp)

        # --------------------------------------------------
        # 2. 新CT -> DeepFluoro subject01
        # --------------------------------------------------
        T_DeepFluoro_from_newCT = np.linalg.inv(
            T_newCT_from_DeepFluoro
        )

        # --------------------------------------------------
        # 3. 新CT -> WBCT foundation reference
        # --------------------------------------------------
        T_WBCT_reference_from_newCT = (
            T_WBCT_reference_from_DeepFluoro
            @ T_DeepFluoro_from_newCT
        )

        write_affine_matrix(
            T_WBCT_reference_from_newCT,
            warp,
        )

        # --------------------------------------------------
        # 4. 用 XVR 自己的 get_4x4 做验收
        # --------------------------------------------------
        T_xvr_reframe = get_4x4(
            str(warp),
            str(new_ct),
            invert=False,
        )

        M = (
            T_xvr_reframe.matrix
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        R = M[:3, :3]
        t = M[:3, 3]

        det_r = float(np.linalg.det(R))
        ortho_error = float(
            np.linalg.norm(R.T @ R - np.eye(3))
        )
        translation_norm = float(
            np.linalg.norm(t)
        )

        status = (
            "OK"
            if abs(det_r - 1.0) < 1e-3
            and ortho_error < 1e-3
            else "CHECK"
        )

        rows.append({
            "case": case_id,
            "new_ct": str(new_ct),
            "deepfluoro_to_newct_warp": str(deepfluoro_to_newct_warp),
            "warp": str(warp),
            "det_R": det_r,
            "orthogonality_error": ortho_error,
            "translation_norm_mm": translation_norm,
            "status": status,
            "error": "",
        })

        print(
            f"  {status}: "
            f"det(R)={det_r:.6f}, "
            f"ortho_err={ortho_error:.3e}, "
            f"|t|={translation_norm:.2f} mm"
        )

    except Exception as e:
        rows.append({
            "case": case_id,
            "new_ct": str(new_ct),
            "deepfluoro_to_newct_warp": str(deepfluoro_to_newct_warp),
            "warp": str(warp),
            "det_R": "",
            "orthogonality_error": "",
            "translation_norm_mm": "",
            "status": "FAILED",
            "error": repr(e),
        })

        print("  FAILED:", repr(e))


with REPORT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case",
            "new_ct",
            "deepfluoro_to_newct_warp",
            "warp",
            "det_R",
            "orthogonality_error",
            "translation_norm_mm",
            "status",
            "error",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n================================")
print("BATCH COMPLETE")
print("================================")
print("Report:", REPORT)
```

运行：

```bash
python scripts/build_wbct_warps.py
```

检查：

```bash
cat data/spine_cases/warp_batch_report.csv
```

---

## 8. 批量处理建议

大规模处理时，不建议仅因为：

```text
det(R) ≈ 1
```

就认为所有 warp 都正确。

建议至少执行三层 QC：

### 第一层：程序级 QC

检查：

```text
warp.mat 是否生成
det(R)
R^T R
是否存在 NaN / Inf
```

### 第二层：统计级 QC

检查：

```text
translation norm
rotation magnitude
```

如果某个病例明显偏离整个数据集分布，应单独检查。

例如绝大多数病例：

```text
|t| ≈ 1000~1300 mm
```

但突然出现：

```text
|t| = 5000 mm
```

这种情况应自动标记为异常，而不是直接进入训练。

注意：具体阈值不要预先写死，应先用一批成功病例统计正常分布后再确定。

### 第三层：可视化 QC

建议：

- 前 20~50 个病例全部查看；
- 管线稳定后随机抽查 5%~10%；
- 所有统计异常病例必须人工检查。

---

## 9. 非常重要：批量生成 warp 与“多 CT 一次训练”是两件事

当前 XVR 原始实现中的：

```python
self.reframe = initialize_coordinate_frame(
    warp,
    volpath,
    invert,
)
```

是在 `Trainer.__init__()` 中只初始化一次。

因此当前 CLI：

```bash
xvr train \
  -v <多个CT的目录> \
  -w <一个warp>
```

本质上只支持一个全局 `warp`，**不能自动根据当前抽到的 CT 选择该 CT 自己的 warp**。

所以：

> 批量生成每个病例的 `warp.mat` 是必要的，但 stock XVR 目前不能直接把这些 per-patient warp 在一个 multi-CT training run 里动态使用。

如果后续准备做真正的大规模脊柱 finetune，推荐修改 XVR：

```text
Subject
├── volume
├── mask
└── warp
```

DataLoader 每次抽取一个 Subject 时，同时加载该病例自己的 warp，然后在当前 iteration 中计算：

```python
reframe = get_4x4(
    subject_warp,
    subject_volume,
    invert=False,
)
```

而不是：

```python
self.reframe = ...
```

只在训练开始时计算一次。

另一个可选方案是：

> 先把全部 CT 物理 resample 到统一 reference frame，然后再进行 multi-volume training。

但这会修改训练输入本身，并涉及 mask 重采样，因此与“保留原 CT + 每病例动态 reframe”相比侵入更大。

对于后续的大规模脊柱微调，优先推荐 **per-subject warp 动态 reframe**。

---

## 10. 当前方案的限制

当前固定用于桥接的是：

```text
DeepFluoro subject01
```

它适合腰椎/骨盆附近数据，因为解剖覆盖较接近。

如果后续数据包含：

- 颈椎
- 上胸椎
- 极短 FOV CT
- 严重畸形脊柱
- 术后金属植入病例

则单一 DeepFluoro anchor 的刚体配准可能不稳定。

因此大规模训练前建议：

1. 先用 20~50 个病例做 pilot；
2. 对所有 bridge registration 做 QC；
3. 如果颈椎/胸椎失败率高，应考虑按解剖区域建立不同 anchor；
4. 不要把配准失败的 warp 静默带入训练。

---

## 11. 文件命名建议

虽然作者脚本中有时使用：

```text
warp.txt
```

而当前下载得到的是：

```text
warp.mat
```

XVR 实际读取的是 ANTs / SimpleITK transform，关键是文件内容能够被：

```python
ants.read_transform(...)
```

正确读取，而不是扩展名本身。

当前已经验证 `.mat` 工作正常，因此批量流程建议统一使用：

```text
warp.mat
deepfluoro_to_newct.mat
```

避免混用。

---

## 12. 当前已验证成功的 1001 流程

目前 `1001` 已验证：

```text
DeepFluoro subject01:
experiments/data/template_bridge/deepfluoro_subject01/
├── volume.nii.gz
└── warp.mat

新CT（本例为 1001）:
data/vertebra/1001/
├── volume.nii.gz
├── bridge_df_to_1001.mat
└── warp.mat
```

其中：

```text
bridge_df_to_1001.mat
```

满足：


\det(R)\approx1


且：


R^TR\approx I


最终：

```text
data/vertebra/1001/warp.mat
```

能够被 XVR `get_4x4()` 正常读取，并成功用于 WBCT foundation checkpoint 的 AP 500-step finetune。

---

# 最终流程总结

对于任意一个新的脊柱 CT，完整关系直接写成：

```text
作者已经提供：
DeepFluoro subject01 CT
+
DeepFluoro subject01 -> WBCT foundation reference 的 warp.mat
                         │
                         ▼
对新的脊柱 CT 执行 ANTs rigid registration

fixed  = 新CT
moving = DeepFluoro subject01 CT
                         │
                         ▼
得到：
DeepFluoro subject01 -> 新CT
                         │
                         ▼
对这个变换取逆，得到：
新CT -> DeepFluoro subject01
                         │
                         ▼
再与作者提供的 warp.mat 组合，得到：
新CT -> WBCT foundation reference
                         │
                         ▼
保存为该新CT自己的 warp.mat
                         │
                         ▼
使用 XVR get_4x4() 验证
                         │
                         ▼
进行配准 QC
                         │
                         ▼
用于 WBCT foundation finetune
```

核心公式直接写完整名称：


\boxed{
T_{\text{WBCT foundation reference}\leftarrow\text{新CT}}
=
T_{\text{WBCT foundation reference}\leftarrow\text{DeepFluoro subject01}}
\left(
T_{\text{新CT}\leftarrow\text{DeepFluoro subject01}}
\right)^{-1}
}


以后不要再把这三个坐标系缩写成 `F / A / P`。文档、代码变量名和调试输出都直接使用完整对象名。
