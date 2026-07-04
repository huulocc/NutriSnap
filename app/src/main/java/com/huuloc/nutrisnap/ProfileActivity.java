package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.widget.ArrayAdapter;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class ProfileActivity extends AppCompatActivity {

    // Hệ số vận động tương ứng với Spinner
    private static final double[] ACTIVITY_MULTIPLIERS = {1.2, 1.375, 1.55, 1.725};

    private FirebaseFirestore db;
    private FirebaseUser currentUser;
    private TextView txtName, txtEmail, txtBmi, txtBmiStatus;
    private TextView txtGoalCalories, txtGoalProtein;
    private TextInputEditText inputHeight, inputWeight, inputAge;
    private RadioGroup radioGender;
    private Spinner spinnerActivity;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_profile);

        currentUser = FirebaseAuth.getInstance().getCurrentUser();
        if (currentUser == null) {
            startActivity(new Intent(this, LoginActivity.class));
            finish();
            return;
        }

        db = FirebaseFirestore.getInstance();

        // Bind views
        txtName = findViewById(R.id.txtName);
        txtEmail = findViewById(R.id.txtEmail);
        txtBmi = findViewById(R.id.txtBmi);
        txtBmiStatus = findViewById(R.id.txtBmiStatus);
        txtGoalCalories = findViewById(R.id.txtGoalCalories);
        txtGoalProtein = findViewById(R.id.txtGoalProtein);
        inputHeight = findViewById(R.id.inputHeight);
        inputWeight = findViewById(R.id.inputWeight);
        inputAge = findViewById(R.id.inputAge);
        radioGender = findViewById(R.id.radioGender);
        spinnerActivity = findViewById(R.id.spinnerActivity);
        MaterialButton btnSave = findViewById(R.id.btnSave);
        MaterialButton btnLogout = findViewById(R.id.btnLogout);

        // Setup Spinner
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.activity_levels, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerActivity.setAdapter(adapter);

        loadProfile();

        btnSave.setOnClickListener(v -> saveProfile());
        btnLogout.setOnClickListener(v -> {
            FirebaseAuth.getInstance().signOut();
            Intent intent = new Intent(this, LoginActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
            finish();
        });

        // Bottom Navigation
        BottomNavigationView bottomNav = findViewById(R.id.bottomNav);
        bottomNav.setSelectedItemId(R.id.nav_profile);
        bottomNav.setOnItemSelectedListener(item -> {
            int id = item.getItemId();
            if (id == R.id.nav_home) {
                startActivity(new Intent(this, HomeActivity.class));
                finish();
                return true;
            }
            if (id == R.id.nav_log) {
                startActivity(new Intent(this, MealLogActivity.class));
                finish();
                return true;
            }
            return true;
        });
    }

    private void loadProfile() {
        txtEmail.setText(currentUser.getEmail());
        db.collection("users").document(currentUser.getUid()).get()
                .addOnSuccessListener(document -> {
                    String name = document.getString("name");
                    Double height = document.getDouble("heightCm");
                    Double weight = document.getDouble("weightKg");
                    Double bmi = document.getDouble("bmi");
                    Long age = document.getLong("age");
                    String gender = document.getString("gender");
                    Long activityLevel = document.getLong("activityLevel");
                    Long goalCal = document.getLong("dailyCalorieGoal");
                    Long goalPro = document.getLong("dailyProteinGoal");

                    if (name != null && !name.isEmpty()) {
                        txtName.setText(name);
                    }
                    if (height != null && height > 0) {
                        inputHeight.setText(formatNumber(height));
                    }
                    if (weight != null && weight > 0) {
                        inputWeight.setText(formatNumber(weight));
                    }
                    if (age != null && age > 0) {
                        inputAge.setText(String.valueOf(age));
                    }
                    if ("female".equals(gender)) {
                        radioGender.check(R.id.radioFemale);
                    } else {
                        radioGender.check(R.id.radioMale);
                    }
                    if (activityLevel != null && activityLevel >= 0 && activityLevel < ACTIVITY_MULTIPLIERS.length) {
                        spinnerActivity.setSelection(activityLevel.intValue());
                    }
                    if (bmi != null && bmi > 0) {
                        showBmi(bmi);
                    }
                    if (goalCal != null && goalCal > 0) {
                        txtGoalCalories.setText(String.valueOf(goalCal));
                    }
                    if (goalPro != null && goalPro > 0) {
                        txtGoalProtein.setText(goalPro + " g");
                    }
                })
                .addOnFailureListener(error ->
                        Toast.makeText(this, error.getMessage(), Toast.LENGTH_SHORT).show());
    }

    private void saveProfile() {
        String heightText = getText(inputHeight);
        String weightText = getText(inputWeight);
        String ageText = getText(inputAge);

        if (TextUtils.isEmpty(heightText) || TextUtils.isEmpty(weightText) || TextUtils.isEmpty(ageText)) {
            Toast.makeText(this, "Vui lòng nhập đầy đủ thông tin", Toast.LENGTH_SHORT).show();
            return;
        }

        double heightCm = Double.parseDouble(heightText);
        double weightKg = Double.parseDouble(weightText);
        int age = Integer.parseInt(ageText);
        int activityIndex = spinnerActivity.getSelectedItemPosition();
        boolean isMale = radioGender.getCheckedRadioButtonId() == R.id.radioMale;

        // Tính BMI
        double heightM = heightCm / 100.0;
        double bmi = weightKg / (heightM * heightM);

        // Tính BMR (Mifflin-St Jeor)
        double bmr;
        if (isMale) {
            bmr = 10 * weightKg + 6.25 * heightCm - 5 * age + 5;
        } else {
            bmr = 10 * weightKg + 6.25 * heightCm - 5 * age - 161;
        }

        // Tính TDEE
        double tdee = bmr * ACTIVITY_MULTIPLIERS[activityIndex];
        long dailyCalorieGoal = Math.round(tdee);

        // Protein mục tiêu: 1.6g / kg thể trọng
        long dailyProteinGoal = Math.round(weightKg * 1.6);

        // Lưu lên Firestore
        Map<String, Object> data = new HashMap<>();
        data.put("heightCm", heightCm);
        data.put("weightKg", weightKg);
        data.put("age", age);
        data.put("gender", isMale ? "male" : "female");
        data.put("activityLevel", activityIndex);
        data.put("bmi", bmi);
        data.put("dailyCalorieGoal", dailyCalorieGoal);
        data.put("dailyProteinGoal", dailyProteinGoal);

        db.collection("users").document(currentUser.getUid())
                .update(data)
                .addOnSuccessListener(unused -> {
                    showBmi(bmi);
                    txtGoalCalories.setText(String.valueOf(dailyCalorieGoal));
                    txtGoalProtein.setText(dailyProteinGoal + " g");
                    Toast.makeText(this, "Đã lưu hồ sơ", Toast.LENGTH_SHORT).show();
                })
                .addOnFailureListener(error ->
                        Toast.makeText(this, error.getMessage(), Toast.LENGTH_SHORT).show());
    }

    private void showBmi(double bmi) {
        txtBmi.setText(String.format(Locale.US, "%.1f", bmi));
        if (bmi < 18.5) {
            txtBmiStatus.setText("Thiếu cân");
        } else if (bmi < 25) {
            txtBmiStatus.setText("Bình thường");
        } else if (bmi < 30) {
            txtBmiStatus.setText("Thừa cân");
        } else {
            txtBmiStatus.setText("Béo phì");
        }
    }

    private String getText(TextInputEditText input) {
        return input.getText() == null ? "" : input.getText().toString().trim();
    }

    private String formatNumber(double value) {
        if (value == Math.rint(value)) {
            return String.valueOf((int) value);
        }
        return String.valueOf(value);
    }
}
