package com.huuloc.nutrisnap;

import android.app.Service;
import android.content.Intent;
import android.net.Uri;
import android.os.Binder;
import android.os.IBinder;

import androidx.annotation.Nullable;

public class FoodRecognitionService extends Service {

    private final IBinder binder = new LocalBinder();

    public class LocalBinder extends Binder {
        FoodRecognitionService getService() {
            return FoodRecognitionService.this;
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    public void recognize(Uri imageUri, PredictionApi.Callback callback) {
        PredictionApi.predict(this, imageUri, callback);
    }
}
