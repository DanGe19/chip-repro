"""选择性下载 TRUMANS Data_release（gdown 版，匿名访问）。

注意（2026-11 实测）：匿名 gdown 下载约 20 个文件后会被 Google Drive 限流
（"had many accesses"）。稳定的做法是用 rclone 加自己的 Google 账号授权：

    # Mac 上拿 token
    rclone authorize "drive"
    # Pod 上写配置（token 换成上一步的 JSON）
    cat > ~/.config/rclone/rclone.conf <<'EOF'
    [gdrive]
    type = drive
    scope = drive.readonly
    root_folder_id = 1CPw_cQcQAP_wbMuQ89rREQmMtEMd-2Hz
    token = {...}
    EOF
    rclone copy gdrive: /data/datasets/trumans \
      --include "/*.npy" --include "/Scene_mesh/**" --include "/Object_all/**" --include "/Actions/**" \
      --transfers 8 --drive-chunk-size 64M --stats 30s --log-level INFO --log-file rclone_trumans.log

本脚本保留作为小规模测试或 rclone 不可用时的备选。
"""
import os
import sys

import gdown

FOLDER = "https://drive.google.com/drive/folders/1CPw_cQcQAP_wbMuQ89rREQmMtEMd-2Hz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/datasets/trumans"
KEEP_PREFIX = ("Scene_mesh/", "Object_all/", "Actions/")
SKIP_PREFIX = ("video_render/", "Recordings_blend/", "Scene_occ_render/", "smplx_result", "Scene/")


def main():
    files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
    todo = [f for f in files if not f.path.startswith(SKIP_PREFIX) and ("/" not in f.path or f.path.startswith(KEEP_PREFIX))]
    print(f"{len(files)} files total, downloading {len(todo)}")
    os.makedirs(OUT, exist_ok=True)
    for i, f in enumerate(todo, 1):
        dst = os.path.join(OUT, f.path)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"[{i}/{len(todo)}] {f.path}", flush=True)
        gdown.download(id=f.id, output=dst, quiet=True)
    print("done")


if __name__ == "__main__":
    main()
