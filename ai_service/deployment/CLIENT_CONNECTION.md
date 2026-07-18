# NutriSnap AI Service Client Connection

## Base URL

Internal/private server URL: `http://172.16.64.21:8000/`

Candidate public URL after firewall/NAT allows TCP 8000: `http://103.130.211.150:8000/`

Current note: the service is listening on `0.0.0.0:8000`, but an independent client test to `103.130.211.150:8000` failed with connection refused. The host/provider firewall or NAT needs to allow TCP 8000 before outside clients can connect directly.

## Health

`GET /health`

## Model

`GET /api/v1/model`

## Predict

`POST /predict`

Content-Type: `multipart/form-data`

## Header

No header is required for the simplified `/predict` endpoint.

## Curl example

```bash
curl -X POST \
  -F "file=@meal.jpg" \
  http://103.130.211.150:8000/predict
```

## Android/Retrofit contract

```java
@Multipart
@POST("predict")
Call<PredictionResponse> predict(
    @Part MultipartBody.Part file
);
```

## Firestore mapping

`response.prediction.model_label -> foods/{model_label}`

## HTTP warning

HTTP does not encrypt uploaded food images or API credentials. Do not use plain HTTP for production Internet deployment.

Future production path: Client -> HTTPS 443 -> Nginx/Caddy -> 127.0.0.1:8000.
