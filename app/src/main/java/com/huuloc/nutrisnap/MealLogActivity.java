package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import java.util.ArrayList;
import java.util.List;

public class MealLogActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_meal_log);

        RecyclerView recycler = findViewById(R.id.recyclerMeals);
        recycler.setLayoutManager(new LinearLayoutManager(this));
        recycler.setAdapter(new MealAdapter(sampleMeals()));

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

    private List<Meal> sampleMeals() {
        List<Meal> meals = new ArrayList<>();
        meals.add(new Meal("Pho Bo", "Khẩu phần: Vừa • 07:30", "420 kcal", "25 g"));
        meals.add(new Meal("Banh Mi", "Khẩu phần: Vừa • 09:00", "350 kcal", "14 g"));
        meals.add(new Meal("Com Tam", "Khẩu phần: Lớn • 12:15", "650 kcal", "32 g"));
        meals.add(new Meal("Bun Bo", "Khẩu phần: Vừa • 15:00", "480 kcal", "28 g"));
        meals.add(new Meal("Goi Cuon", "Khẩu phần: Nhỏ • 18:40", "180 kcal", "11 g"));
        return meals;
    }
}
