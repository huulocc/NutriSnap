# NutriSnap AI Mock/Manual Test Commands

API key is stored on the server in `ai_service/.env.production`. Do not commit it or paste it in chat.

## Local Server Health

```bash
curl http://127.0.0.1:8000/health
```

## Authenticated Readiness/Model Checks

```bash
cd /data/quyhv/data/NutriSnap
export NUTRISNAP_API_KEY=$(grep '^API_KEY=' ai_service/.env.production | cut -d= -f2-)

curl -H "X-API-Key: $NUTRISNAP_API_KEY" \
  http://127.0.0.1:8000/ready

curl -H "X-API-Key: $NUTRISNAP_API_KEY" \
  http://127.0.0.1:8000/api/v1/model
```

## Real-Image Prediction Mock Test

Use any jpg/png/webp. This example uses one validation image only for integration testing, not accuracy reporting.

```bash
curl -X POST \
  -F "file=@/data/quyhv/data/NutriSnap-work/session-20260717-214426-7J1KGz/repo/ai_training/data/source/Images/Validate/Pho/16.jpg;type=image/jpeg" \
  http://127.0.0.1:8000/predict
```

## Outside-Client Command Once TCP 8000 Is Reachable

```bash
curl -X POST \
  -F "file=@meal.jpg" \
  http://103.130.211.150:8000/predict
```

Expected response schema is saved in `ai_service/deployment/sample_predict_response.json`.
