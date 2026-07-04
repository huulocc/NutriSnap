package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
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

        ImageView btnBack = findViewById(R.id.btnBack);
        MaterialButton btnSaveMeal = findViewById(R.id.btnSaveMeal);

        btnBack.setOnClickListener(v -> finish());
        btnSaveMeal.setOnClickListener(v -> saveMeal(btnSaveMeal));
    }

    private void saveMeal(MaterialButton btnSaveMeal) {
        Date now = new Date();
        String dateKey = new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(now);
        String mealTime = new SimpleDateFormat("HH:mm", Locale.US).format(now);

        Map<String, Object> meal = new HashMap<>();
        meal.put("foodId", "pho_bo");
        meal.put("foodName", "Pho Bo");
        meal.put("displayName", "Pho Bo");
        meal.put("portion", "Medium");
        meal.put("calories", 420);
        meal.put("protein", 25);
        meal.put("carbs", 55);
        meal.put("fat", 12);
        meal.put("confidence", 0.95);
        meal.put("imageUrl", "");
        meal.put("dateKey", dateKey);
        meal.put("mealTime", mealTime);
        meal.put("eatenAt", FieldValue.serverTimestamp());

        btnSaveMeal.setEnabled(false);
        db.collection("users")
                .document(currentUser.getUid())
                .collection("mealLogs")
                .add(meal)
                .addOnSuccessListener(document -> {
                    Toast.makeText(this, "Da luu bua an", Toast.LENGTH_SHORT).show();
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
