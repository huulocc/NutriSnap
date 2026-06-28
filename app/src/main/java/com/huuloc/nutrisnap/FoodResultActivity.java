package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

public class FoodResultActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_food_result);

        ImageView btnBack = findViewById(R.id.btnBack);
        MaterialButton btnSaveMeal = findViewById(R.id.btnSaveMeal);

        btnBack.setOnClickListener(v -> finish());
        btnSaveMeal.setOnClickListener(v -> {
            Intent intent = new Intent(this, MealLogActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });
    }
}
