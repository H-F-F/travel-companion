# Backend Quick Start

## Option A (Recommended): Zero-dependency server
No third-party package required.

```powershell
python -m backend.simple_server
```

Open UI: `http://127.0.0.1:8000/`

## AMap integration (recommended)
```powershell
$env:AMAP_API_KEY="your_amap_web_key"
python -m backend.simple_server
```

Or put the key in the repo root `.env` file:
```env
AMAP_API_KEY=your_amap_web_key
MCP_12306_URL=http://127.0.0.1:8001/mcp
AMAP_NEAREST_RAIL_FIRST=0
```

Enabled features with AMap key:
- Forward geocoding for address -> latitude/longitude
- Reverse geocoding for location -> province/city/county
- Address search (`/api/v1/search-address`)
- Distance matrix for route leg distance/time
- Weather (live weather from AMap)
- 3-day forecast panel (AMap preferred, fallback forecast supported)
- Independent weather endpoint (`/api/v1/weather?address=<地址>&days=3`)
- Candidate-based planning fields on `POST /api/v1/plan`
- Attraction theme filtering (`attraction_styles`) and paged attraction browsing on frontend
- POI search endpoint (`/api/v1/search-poi`) for attraction / food search, with attraction results grouped by AMap `typecode`
- 12306 station search endpoint (`/api/v1/transport/stations`)
- 12306 ticket / transfer lookup embedded into `POST /api/v1/plan`
- Optional nearest-rail prioritization for destination hub lookup (`AMAP_NEAREST_RAIL_FIRST=1` to enable, `0` to disable)

## 12306 MCP integration (optional)
Configure the MCP endpoint in `.env` or your shell before starting the backend:

```powershell
$env:MCP_12306_URL="http://127.0.0.1:8001/mcp"
python -m backend.simple_server
```

Recommended: bind the 12306 MCP container/service to host port `8001` or another port that is not `8000`, because this project already uses `8000` by default.
If the repo root already contains a local `.mcp12306/node_modules/12306-mcp`, the backend will also auto-detect it and call the package through local `stdio`.

Geocode example:
```powershell
curl "http://127.0.0.1:8000/api/v1/geocode?address=汉中&city=汉中"
```

Weather example:
```powershell
curl "http://127.0.0.1:8000/api/v1/weather?address=汉中&days=3"
```

## Option B: FastAPI server
```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Test request
```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/plan" `
  -H "Content-Type: application/json" `
  -d '{"city":"Shanghai","travel_date":"2026-03-14","days":2,"planning_mode":"candidate","booking_date":"2026-03-10","origin_city":"Xi\\u0027an","origin_station":"Xi\\u0027an North","budget_level":"standard","food_preferences":["spicy","local"],"selected_attractions":["The Bund"],"excluded_foods":["Dessert Cafe"],"live_poi":true,"location_lat":31.2304,"location_lon":121.4737,"pace":"normal"}'
```

`live_poi`:
- `false`：仅本地样本，响应更快
- `true`：尝试实时POI，超时自动回退

Planning additions:
- `planning_mode=auto`：直接按系统推荐生成路线
- `planning_mode=candidate`：返回扩展推荐列表、预计天数、提醒信息，并允许前端在推荐区直接二次筛选
- `attraction_styles`：按人文 / 自然 / 地标 / 室内标签提升景点排序优先级
- `selected_attractions` / `selected_foods`：优先保留的候选项
- `excluded_attractions` / `excluded_foods`：本轮路线中排除的候选项
- `booking_date` / `origin_city` / `origin_station`：会触发 12306 车票和中转查询；未配置 `MCP_12306_URL` 时自动降级
