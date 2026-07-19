package com.huuloc.nutrisnap;

import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.ChipGroup;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FieldValue;
import com.google.firebase.firestore.FirebaseFirestore;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class FoodResultActivity extends AppCompatActivity {

    private FirebaseFirestore db;
    private FirebaseUser currentUser;

    private String imageUri;
    private String modelLabel;
    private String displayName;
    private double confidence;
    private int baseCalories;
    private int baseProtein;

    private TextView txtCalories;
    private TextView txtProtein;
    private ChipGroup portionGroup;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_food_result);

        currentUser = FirebaseAuth.getInstance().getCurrentUser();
        if (currentUser == null) {
            startActivity(new Intent(this, LoginActivity.class));
            finish();
            return;
        }

        db = FirebaseFirestore.getInstance();

        imageUri = getIntent().getStringExtra("imageUri");
        modelLabel = getIntent().getStringExtra("modelLabel");
        displayName = getIntent().getStringExtra("displayName");
        confidence = getIntent().getDoubleExtra("confidence", 0);

        FoodNutrition.Info fallback = FoodNutrition.get(modelLabel);
        baseCalories = fallback.calories;
        baseProtein = fallback.protein;

        ImageView btnBack = findViewById(R.id.btnBack);
        ImageView imageFood = findViewById(R.id.imageFood);
        TextView txtFoodName = findViewById(R.id.txtFoodName);
        TextView txtConfidence = findViewById(R.id.txtConfidence);
        txtCalories = findViewById(R.id.txtCalories);
        txtProtein = findViewById(R.id.txtProtein);
        portionGroup = findViewById(R.id.portionGroup);
        MaterialButton btnSaveMeal = findViewById(R.id.btnSaveMeal);

        txtFoodName.setText(displayName);
        txtConfidence.setText("Độ tin cậy: " + Math.round(confidence * 100) + "%");

        if (imageUri != null) {
            Glide.with(this).load(Uri.parse(imageUri)).into(imageFood);
        }

        updateNutrition();
        portionGroup.setOnCheckedStateChangeListener((group, ids) -> updateNutrition());
        loadNutritionFromProvider();

        btnBack.setOnClickListener(v -> finish());
        btnSaveMeal.setOnClickListener(v -> saveMeal(btnSaveMeal));
    }

    private void loadNutritionFromProvider() {
        Handler mainHandler = new Handler(Looper.getMainLooper());
        new Thread(() -> {
            Uri uri = Uri.withAppendedPath(NutritionProvider.FOODS_URI, modelLabel);
            Cursor cursor = getContentResolver().query(uri, null, null, null, null);
            if (cursor != null) {
                if (cursor.moveToFirst()) {
                    int calories = cursor.getInt(cursor.getColumnIndexOrThrow(NutritionProvider.COL_CALORIES));
                    int protein = cursor.getInt(cursor.getColumnIndexOrThrow(NutritionProvider.COL_PROTEIN));
                    if (calories > 0) {
                        mainHandler.post(() -> {
                            baseCalories = calories;
                            baseProtein = protein;
                            updateNutrition();
                        });
                    }
                }
                cursor.close();
            }
        }).start();
    }

    private double portionMultiplier() {
        int checkedId = portionGroup.getCheckedChipId();
        if (checkedId == R.id.chipSmall) return 0.5;
        if (checkedId == R.id.chipLarge) return 1.5;
        if (checkedId == R.id.chipXLarge) return 2.0;
        return 1.0;
    }

    private String portionName() {
        int checkedId = portionGroup.getCheckedChipId();
        if (checkedId == R.id.chipSmall) return "Nhỏ";
        if (checkedId == R.id.chipLarge) return "Lớn";
        if (checkedId == R.id.chipXLarge) return "Rất lớn";
        return "Vừa";
    }

    private void updateNutrition() {
        double multiplier = portionMultiplier();
        txtCalories.setText(String.valueOf((int) Math.round(baseCalories * multiplier)));
        txtProtein.setText(String.valueOf((int) Math.round(baseProtein * multiplier)));
    }

    private void saveMeal(MaterialButton btnSaveMeal) {
        double multiplier = portionMultiplier();
        long calories = Math.round(baseCalories * multiplier);
        long protein = Math.round(baseProtein * multiplier);

        Date now = new Date();
        String dateKey = new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(now);
        String mealTime = new SimpleDateFormat("HH:mm", Locale.US).format(now);

        Map<String, Object> meal = new HashMap<>();
        meal.put("foodId", modelLabel);
        meal.put("foodName", modelLabel);
        meal.put("displayName", displayName);
        meal.put("portion", portionName());
        meal.put("calories", calories);
        meal.put("protein", protein);
        meal.put("confidence", confidence);
        meal.put("imageUrl", imageUri == null ? "" : imageUri);
        meal.put("dateKey", dateKey);
        meal.put("mealTime", mealTime);
        meal.put("eatenAt", FieldValue.serverTimestamp());

        btnSaveMeal.setEnabled(false);
        db.collection("users")
                .document(currentUser.getUid())
                .collection("mealLogs")
                .add(meal)
                .addOnSuccessListener(document -> {
                    Toast.makeText(this, "Đã lưu bữa ăn", Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(this, MealLogActivity.class);
                    intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(intent);
                    finish();
                })
                .addOnFailureListener(error -> {
                    btnSaveMeal.setEnabled(true);
                    Toast.makeText(this, error.getMessage(), Toast.LENGTH_SHORT).show();
                });
    }
}
