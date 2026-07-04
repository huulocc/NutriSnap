package com.huuloc.nutrisnap;

import android.os.Bundle;
import android.text.TextUtils;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.firestore.FieldValue;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Map;

public class RegisterActivity extends AppCompatActivity {

    private FirebaseAuth auth;
    private FirebaseFirestore db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        auth = FirebaseAuth.getInstance();
        db = FirebaseFirestore.getInstance();

        TextInputEditText inputName = findViewById(R.id.inputName);
        TextInputEditText inputEmail = findViewById(R.id.inputEmail);
        TextInputEditText inputPassword = findViewById(R.id.inputPassword);
        TextInputEditText inputConfirm = findViewById(R.id.inputConfirm);
        MaterialButton btnRegister = findViewById(R.id.btnRegister);
        TextView txtGoLogin = findViewById(R.id.txtGoLogin);

        btnRegister.setOnClickListener(v -> {
            String name = getText(inputName);
            String email = getText(inputEmail);
            String password = getText(inputPassword);
            String confirm = getText(inputConfirm);

            if (TextUtils.isEmpty(name) || TextUtils.isEmpty(email) || TextUtils.isEmpty(password)) {
                Toast.makeText(this, "Nhap day du thong tin", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!password.equals(confirm)) {
                Toast.makeText(this, "Mat khau xac nhan khong khop", Toast.LENGTH_SHORT).show();
                return;
            }

            btnRegister.setEnabled(false);
            auth.createUserWithEmailAndPassword(email, password)
                    .addOnSuccessListener(result -> {
                        String uid = result.getUser().getUid();
                        Map<String, Object> user = new HashMap<>();
                        user.put("name", name);
                        user.put("email", email);
                        user.put("heightCm", 0);
                        user.put("weightKg", 0);
                        user.put("bmi", 0);
                        user.put("dailyCalorieGoal", 2000);
                        user.put("dailyProteinGoal", 90);
                        user.put("createdAt", FieldValue.serverTimestamp());

                        db.collection("users").document(uid).set(user)
                                .addOnSuccessListener(unused -> {
                                    Toast.makeText(this, "Dang ky thanh cong", Toast.LENGTH_SHORT).show();
                                    finish();
                                })
                                .addOnFailureListener(error -> {
                                    btnRegister.setEnabled(true);
                                    Toast.makeText(this, error.getMessage(), Toast.LENGTH_SHORT).show();
                                });
                    })
                    .addOnFailureListener(error -> {
                        btnRegister.setEnabled(true);
                        Toast.makeText(this, error.getMessage(), Toast.LENGTH_SHORT).show();
                    });
        });
        txtGoLogin.setOnClickListener(v -> finish());
    }

    private String getText(TextInputEditText input) {
        return input.getText().toString().trim();
    }
}
