package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;

import androidx.appcompat.app.AppCompatActivity;

import com.google.firebase.auth.FirebaseAuth;

public class SplashActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            Class<?> nextScreen = FirebaseAuth.getInstance().getCurrentUser() == null
                    ? LoginActivity.class
                    : HomeActivity.class;
            startActivity(new Intent(this, nextScreen));
            finish();
        }, 1000);
    }
}
