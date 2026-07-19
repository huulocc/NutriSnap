package com.huuloc.nutrisnap;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.progressindicator.CircularProgressIndicator;
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

public class HomeActivity extends AppCompatActivity {

    private FirebaseFirestore db;
    private FirebaseUser currentUser;
    private RecyclerView recycler;
    private TextView txtUserName;
    private TextView txtCalories;
    private TextView txtProtein;
    private CircularProgressIndicator progressCalories;
    private CircularProgressIndicator progressProtein;
    
    private long dailyCalorieGoal = 2000;
    private long dailyProteinGoal = 100;
    private long currentCalories = 0;
    private long currentProtein = 0;

    private final ActivityResultLauncher<String> notificationPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        currentUser = FirebaseAuth.getInstance().getCurrentUser();
        if (currentUser == null) {
            startActivity(new Intent(this, LoginActivity.class));
            finish();
            return;
        }

        db = FirebaseFirestore.getInstance();

        txtUserName = findViewById(R.id.txtUserName);
        txtCalories = findViewById(R.id.txtCalories);
        txtProtein = findViewById(R.id.txtProtein);
        progressCalories = findViewById(R.id.progressCalories);
        progressProtein = findViewById(R.id.progressProtein);

        recycler = findViewById(R.id.recyclerMeals);
        recycler.setLayoutManager(new LinearLayoutManager(this));
        recycler.setAdapter(new MealAdapter(new ArrayList<>()));

        loadUser();
        loadTodayMeals();
        setupReminder();

        MaterialButton btnScan = findViewById(R.id.btnScan);
        btnScan.setOnClickListener(v ->
                startActivity(new Intent(this, ScanFoodActivity.class)));

        BottomNavigationView bottomNav = findViewById(R.id.bottomNav);
        bottomNav.setSelectedItemId(R.id.nav_home);
        bottomNav.setOnItemSelectedListener(item -> {
            int id = item.getItemId();
            if (id == R.id.nav_log) {
                startActivity(new Intent(this, MealLogActivity.class));
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
            loadUser();
            loadTodayMeals();
        }
    }

    private void updateProgressUI() {
        txtCalories.setText(currentCalories + "/" + dailyCalorieGoal);
        txtProtein.setText(currentProtein + "/" + dailyProteinGoal);
        
        progressCalories.setMax((int) dailyCalorieGoal);
        progressCalories.setProgress((int) currentCalories, true); // true for animation
        
        progressProtein.setMax((int) dailyProteinGoal);
        progressProtein.setProgress((int) currentProtein, true);
    }

    private void loadUser() {
        db.collection("users").document(currentUser.getUid()).get()
                .addOnSuccessListener(document -> {
                    String name = document.getString("name");
                    if (name != null && !name.isEmpty()) {
                        txtUserName.setText(name);
                    }
                    
                    Long goalCal = document.getLong("dailyCalorieGoal");
                    Long goalPro = document.getLong("dailyProteinGoal");
                    if (goalCal != null && goalCal > 0) dailyCalorieGoal = goalCal;
                    if (goalPro != null && goalPro > 0) dailyProteinGoal = goalPro;
                    
                    updateProgressUI();
                });
    }

    private void setupReminder() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS);
        }
        ReminderScheduler.scheduleDaily(this);
    }

    private void loadTodayMeals() {
        String today = new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(new Date());
        db.collection("users")
                .document(currentUser.getUid())
                .collection("mealLogs")
                .whereEqualTo("dateKey", today)
                .get()
                .addOnSuccessListener(snapshot -> {
                    List<Meal> meals = new ArrayList<>();
                    long totalCalories = 0;
                    long totalProtein = 0;

                    for (DocumentSnapshot doc : snapshot.getDocuments()) {
                        String foodName = doc.getString("displayName");
                        String portion = doc.getString("portion");
                        String imageUrl = doc.getString("imageUrl");
                        
                        String mealTime = "";
                        Timestamp ts = doc.getTimestamp("eatenAt");
                        if (ts != null) {
                            SimpleDateFormat timeFormat = new SimpleDateFormat("HH:mm", Locale.getDefault());
                            mealTime = timeFormat.format(ts.toDate());
                        }

                        long calories = getLong(doc, "calories");
                        long protein = getLong(doc, "protein");

                        totalCalories += calories;
                        totalProtein += protein;
                        
                        meals.add(new Meal(
                                foodName == null ? "Không tên" : foodName,
                                "Khẩu phần: " + safeText(portion) + " - " + mealTime,
                                calories + " kcal",
                                protein + " g",
                                imageUrl
                        ));
                    }

                    currentCalories = totalCalories;
                    currentProtein = totalProtein;
                    updateProgressUI();
                    
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
