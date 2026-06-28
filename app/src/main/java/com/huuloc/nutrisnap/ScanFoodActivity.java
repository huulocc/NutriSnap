package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

public class ScanFoodActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scan_food);

        ImageView btnBack = findViewById(R.id.btnBack);
        MaterialButton btnAnalyze = findViewById(R.id.btnAnalyze);

        btnBack.setOnClickListener(v -> finish());
        btnAnalyze.setOnClickListener(v ->
                startActivity(new Intent(this, FoodResultActivity.class)));
    }
}
