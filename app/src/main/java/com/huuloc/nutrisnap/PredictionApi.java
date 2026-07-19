package com.huuloc.nutrisnap;

import android.content.Context;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class PredictionApi {

    private static final String PREDICT_URL =
            "https://facts-editor-nil-extended.trycloudflare.com/predict";
    private static final String BOUNDARY = "----NutriSnapBoundary";
    private static final String LINE = "\r\n";

    public interface Callback {
        void onSuccess(String modelLabel, String displayName, double confidence);

        void onError(String message);
    }

    public static void predict(Context context, Uri imageUri, Callback callback) {
        Handler mainHandler = new Handler(Looper.getMainLooper());

        new Thread(() -> {
            try {
                // Nén ảnh trước khi gửi
                InputStream is = context.getContentResolver().openInputStream(imageUri);
                android.graphics.Bitmap bitmap = android.graphics.BitmapFactory.decodeStream(is);
                
                // Resize ảnh nếu quá lớn (tối đa 1024px)
                int width = bitmap.getWidth();
                int height = bitmap.getHeight();
                if (width > 1024 || height > 1024) {
                    float scale = Math.min(1024f / width, 1024f / height);
                    bitmap = android.graphics.Bitmap.createScaledBitmap(bitmap, 
                            (int)(width * scale), (int)(height * scale), true);
                }

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 80, baos);
                byte[] imageBytes = baos.toByteArray();

                String responseBody = upload(imageBytes);

                JSONObject json = new JSONObject(responseBody);
                JSONObject prediction = json.getJSONObject("prediction");
                String modelLabel = prediction.getString("model_label");
                String displayName = prediction.getString("display_name");
                double confidence = prediction.getDouble("confidence");

                mainHandler.post(() -> callback.onSuccess(modelLabel, displayName, confidence));
            } catch (Exception e) {
                mainHandler.post(() -> callback.onError("Không thể nhận diện: " + e.getMessage()));
            }
        }).start();
    }

    private static String upload(byte[] imageBytes) throws Exception {
        URL url = new URL(PREDICT_URL);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setDoOutput(true);
        connection.setRequestMethod("POST");
        connection.setConnectTimeout(20000);
        connection.setReadTimeout(20000);
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + BOUNDARY);

        DataOutputStream output = new DataOutputStream(connection.getOutputStream());
        output.writeBytes("--" + BOUNDARY + LINE);
        output.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"food.jpg\"" + LINE);
        output.writeBytes("Content-Type: image/jpeg" + LINE + LINE);
        output.write(imageBytes);
        output.writeBytes(LINE);
        output.writeBytes("--" + BOUNDARY + "--" + LINE);
        output.flush();
        output.close();

        int code = connection.getResponseCode();
        InputStream stream = code == 200 ? connection.getInputStream() : connection.getErrorStream();
        String body = new String(readBytes(stream));
        connection.disconnect();

        if (code != 200) {
            throw new Exception("Máy chủ trả về mã " + code);
        }
        return body;
    }

    private static byte[] readBytes(InputStream inputStream) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        int read;
        while ((read = inputStream.read(chunk)) != -1) {
            buffer.write(chunk, 0, read);
        }
        inputStream.close();
        return buffer.toByteArray();
    }
}
