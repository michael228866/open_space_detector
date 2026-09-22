# free-space-analyzer

以 Depth Map 為核心，將場景還原成地面占用網格，再判斷室內是否具有足夠大的連續活動空間。V1 不依賴 AI 模型，判定結果可量測、可調參，也能說明失敗原因。

## 已完成能力

- Z-depth 與 radial depth 輸入
- 深度清洗與 pinhole 3D 投影
- 已知相機高度或 RANSAC Ground Plane
- `UNKNOWN / FREE / OCCUPIED` 鳥瞰占用網格
- ray casting 與障礙物安全邊界
- 最大連續空地、玩家可達面積（硬性條件只採計玩家走得到的區域）
- 最大軸對齊空矩形（限制在玩家可達範圍內）
- 最近障礙物、玩家 clearance、最大 clearance
- hard rules、0–100 分數與具體失敗原因
- CLI、JSON 結果與 PNG debug 圖
- 合成場景產生器與單元測試

## 安裝

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test]"
```

## 30 秒跑起來

```bash
python tools/generate_synthetic_scene.py --output-dir sample_data
free-space-analyzer sample_data/depth.npy \
  --camera sample_data/camera.json \
  --config configs/default.yaml \
  --output-json output/result.json \
  --debug-png output/occupancy.png
```

Windows PowerShell 可寫成一行：

```powershell
free-space-analyzer sample_data/depth.npy --camera sample_data/camera.json --config configs/default.yaml --output-json output/result.json --debug-png output/occupancy.png
```

## Python API

```python
import numpy as np
from free_space_analyzer import CameraInfo, DepthFrame, FreeSpaceAnalyzer

depth = np.load("depth.npy")
camera = CameraInfo.from_horizontal_fov(
    width=depth.shape[1],
    height=depth.shape[0],
    horizontal_fov_deg=90.0,
    camera_height_m=1.7,
)

analyzer = FreeSpaceAnalyzer.from_config("configs/default.yaml")
result = analyzer.analyze(depth=depth, camera=camera)

# 串流整合也可以把 metadata 綁成一個 frame：
frame = DepthFrame(depth, camera, timestamp_s=0.0, frame_id="capture-0001")
same_result = analyzer.analyze_frame(frame)

print(result.is_open_space)
print(result.score)
print(result.failure_reasons)
print(result.to_dict())
```

## Camera JSON

可以直接提供內參：

```json
{
  "width": 1280,
  "height": 720,
  "fx": 640.0,
  "fy": 640.0,
  "cx": 639.5,
  "cy": 359.5,
  "camera_height_m": 1.7,
  "depth_type": "z_depth",
  "up_vector": [0.0, 1.0, 0.0],
  "player_offset_m": [0.0, 0.0]
}
```

`player_offset_m` 是玩家在地面座標的 `(x, z)`，相對於相機的地面投影：

- 第一人稱：`[0.0, 0.0]`（預設）。
- 第三人稱：角色在相機前方，例如 `[0.0, 3.2]`。跟隨相機的 rig 固定時這是一個
  常數，校準一次即可，不需要逐張匯出。射線起點仍然是相機；改變的只有
  「從哪裡評估空間」：可達區、clearance、debug 藍點。

或使用水平 FOV：

```json
{
  "width": 1280,
  "height": 720,
  "horizontal_fov_deg": 90.0,
  "camera_height_m": 1.7,
  "depth_type": "z_depth"
}
```

`depth_type`：

- `z_depth`：沿相機光軸 Z 的距離。
- `radial`：相機中心到表面的射線距離，程式會轉成 Z-depth。

如果引擎輸出的是 0–1 非線性 depth buffer，必須先用引擎的 projection/near/far 參數轉成公尺；不要直接把 0–1 丟進本程式。

## 結果範例

`sample_data` 合成場景搭配 `configs/default.yaml` 的實際輸出：

```json
{
  "is_open_space": false,
  "score": 73.29,
  "largest_free_area_m2": 54.19,
  "player_reachable_area_m2": 54.19,
  "largest_free_rectangle": {
    "width_m": 2.7,
    "depth_m": 4.1,
    "area_m2": 11.07,
    "x_min_m": -4.7,
    "z_min_m": 4.6,
    "x_max_m": -2.0,
    "z_max_m": 8.7
  },
  "nearest_obstacle_m": 2.75,
  "max_clearance_m": 1.8,
  "obstacle_ratio": 0.1183,
  "unknown_ratio": 0.3398,
  "player_clearance": false,
  "ground_inlier_ratio": 0.62721,
  "ground_method": "known_height",
  "observed_points": 8489,
  "failure_reasons": [
    "required_free_rectangle_not_found",
    "player_clearance_below_minimum"
  ]
}
```

兩個容易誤讀的欄位：

- `player_clearance` 在預設設定下會把玩家旁邊的 UNKNOWN 當成不安全。單張前視畫面
  看不到身體兩側，所以這裡 fail 是正確行為，不是 bug。
- `ground_inlier_ratio` 現在是實際量測值：below-camera 的點裡有多少落在地面平面
  的 `distance_tolerance_m` 內。牆面與障礙物表面也算在分母，所以正確設定也不會是
  1.0；接近 0 才代表 `camera_height_m` 或 `up_vector` 給錯了。

## Debug 圖顏色

- 灰：UNKNOWN
- 白：FREE
- 紅：OCCUPIED（已包含安全邊界）
- 綠框：找到的最大空矩形
- 藍點：玩家在地面的投影位置

## 調參

所有產品門檻都在 `configs/default.yaml`：

- `ground.mode`：已知高度用 `known_height`，需估計地板才用 `ransac`。
- `occupancy.resolution_m`：0.1 代表 10 cm；越小越精細但越慢。
- `occupancy.safety_margin_m`：障礙物外擴半徑。
- `occupancy.player_radius_m`：第三人稱必設（約 0.35）。角色自己的 depth 會落在
  要評估的位置上，不排除掉就等於腳下永遠站著一個假障礙物。第一人稱維持 0。
- `open_space.min_free_area_m2`：最大連續空地門檻。
- `open_space.min_rectangle_*`：必要活動矩形。
- `open_space.max_unknown_ratio`：資訊不足時直接 fail。
- `open_space.nearby_unknown_is_unsafe`：預設 `true`。單張 90° 前視畫面的側邊本來就沒有觀測，玩家旁邊的 UNKNOWN 會直接讓 clearance fail，這是刻意的安全語意；只有離線診斷才建議關掉。

預設 `max_unknown_ratio: 0.45` 是給單張約 90° 前視 Depth 的 V1 起點；若之後加入轉頭掃描或多視角融合，建議逐步收緊到 `0.20`。矩形 hard rule 會容許一個 grid cell 的量化誤差，例如 10 cm 網格量到 3.9 m 可視為滿足 4.0 m 邊界，但輸出的原始量測值不會被改寫。

## 評估擷取點（多機位）

一個候選點一個資料夾，裡面每支相機一個子資料夾。預設會把所有視角融合成一張
以玩家為中心的網格，然後給一個判定：

```bash
python tools/ue_ring_dataset.py <point_dir> --debug-dir output/dbg
python tools/ue_ring_dataset.py <point_dir> --per-view   # 逐機位診斷
```

單一視角看不到角色背後，也看不出自己視錐之外，所以**不要用單張判定**。實測同一
個開闊點：單視角 12 次全部 fail（unknown 0.44–0.77），融合後 unknown 降到 0.01
並正確判為 open。視角數約 8–12 就收斂，再多是餘裕。

成本是線性的，約 0.19 秒/視角 —— 68 支相機的點約 13 秒、230 MB。

## 接 Unreal / Unity 前要確認

1. Depth 是 Z-depth、radial，還是 normalized nonlinear buffer？
2. 單位是 cm 還是 m？
3. 無效/天空深度是 0、1、far plane、Inf 還是 NaN？
4. 使用的是水平或垂直 FOV？
5. Camera height 與 `up_vector` 是否正確？
6. 透明物件、玻璃、遮罩材質是否會寫入 depth？
7. 哪些物件有 depth 但實際不應算碰撞障礙？

完整不變量、模組責任與 Claude Code 接手規則請看 `ARCHITECTURE.md` 與 `AGENTS.md`。
