package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.button.MaterialButton;

import java.util.ArrayList;
import java.util.List;

public class HomeActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        RecyclerView recycler = findViewById(R.id.recyclerMeals);
        recycler.setLayoutManager(new LinearLayoutManager(this));
        recycler.setAdapter(new MealAdapter(sampleMeals()));

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

    private List<Meal> sampleMeals() {
        List<Meal> meals = new ArrayList<>();
        meals.add(new Meal("Pho Bo", "Khẩu phần: Vừa • 07:30", "420 kcal", "25 g"));
        meals.add(new Meal("Com Tam", "Khẩu phần: Lớn • 12:15", "650 kcal", "32 g"));
        meals.add(new Meal("Goi Cuon", "Khẩu phần: Nhỏ • 18:40", "180 kcal", "11 g"));
        return meals;
    }
}
