package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.firebase.Timestamp;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.Query;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MealLogActivity extends AppCompatActivity {

    private FirebaseFirestore db;
    private FirebaseUser currentUser;
    private RecyclerView recycler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_meal_log);

        currentUser = FirebaseAuth.getInstance().getCurrentUser();
        if (currentUser == null) {
            startActivity(new Intent(this, LoginActivity.class));
            finish();
            return;
        }

        db = FirebaseFirestore.getInstance();

        recycler = findViewById(R.id.recyclerMeals);
        recycler.setLayoutManager(new LinearLayoutManager(this));
        recycler.setAdapter(new MealAdapter(new ArrayList<>()));
        loadMealLogs();

        BottomNavigationView bottomNav = findViewById(R.id.bottomNav);
        bottomNav.setSelectedItemId(R.id.nav_log);
        bottomNav.setOnItemSelectedListener(item -> {
            int id = item.getItemId();
            if (id == R.id.nav_home) {
                startActivity(new Intent(this, HomeActivity.class));
                finish();
                return true;
            }
            if (id == R.id.nav_profile) {
                startActivity(new Intent(this, ProfileActivity.class));
                return true;
            }
            return true;
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (currentUser != null && db != null) {
            loadMealLogs();
        }
    }

    private void loadMealLogs() {
        db.collection("users")
                .document(currentUser.getUid())
                .collection("mealLogs")
                .orderBy("eatenAt", Query.Direction.DESCENDING)
                .limit(50)
                .get()
                .addOnSuccessListener(snapshot -> {
                    List<Meal> meals = new ArrayList<>();
                    for (DocumentSnapshot document : snapshot.getDocuments()) {
                        String foodName = document.getString("displayName");
                        String portion = document.getString("portion");
                        String imageUrl = document.getString("imageUrl");
                        
                        String mealTime = "";
                        Timestamp ts = document.getTimestamp("eatenAt");
                        if (ts != null) {
                            SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm", Locale.getDefault());
                            mealTime = sdf.format(ts.toDate());
                        }

                        long calories = getLong(document, "calories");
                        long protein = getLong(document, "protein");

                        meals.add(new Meal(
                                foodName == null ? "Không tên" : foodName,
                                "Khẩu phần: " + safeText(portion) + " - " + mealTime,
                                calories + " kcal",
                                protein + " g",
                                imageUrl
                        ));
                    }
                    recycler.setAdapter(new MealAdapter(meals));
                })
                .addOnFailureListener(error ->
                        Toast.makeText(this, "Lỗi: " + error.getMessage(), Toast.LENGTH_SHORT).show());
    }

    private long getLong(DocumentSnapshot document, String field) {
        Long value = document.getLong(field);
        return value == null ? 0 : value;
    }

    private String safeText(String value) {
        return value == null ? "" : value;
    }
}
