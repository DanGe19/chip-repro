# 展示页（GitHub Pages）

`docs/index.html` 是一个纯静态页面，视频放在 `docs/videos/`（mp4，建议每段 < 3 MB，用
`ffmpeg -i in.mp4 -vf scale=640:-2 -crf 28 -an out.mp4` 压一下）。

发布：GitHub 仓库 → Settings → Pages → Source 选 "Deploy from a branch"，Branch 选 `main`、目录选 `/docs`
。
几分钟后页面在 https://dange19.github.io/chip-repro/ 。

加视频：把 mp4 放进 `videos/`，在 `index.html` 底部的 `LOCO / HOI / POLICY` 列表里加一行即可。
