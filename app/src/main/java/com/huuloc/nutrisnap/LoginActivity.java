package com.huuloc.nutrisnap;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.widget.Toast;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;

public class LoginActivity extends AppCompatActivity {

    private FirebaseAuth auth;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        auth = FirebaseAuth.getInstance();

        TextInputEditText inputEmail = findViewById(R.id.inputEmail);
        TextInputEditText inputPassword = findViewById(R.id.inputPassword);
        MaterialButton btnLogin = findViewById(R.id.btnLogin);
        TextView txtGoRegister = findViewById(R.id.txtGoRegister);

        btnLogin.setOnClickListener(v -> {
            String email = getText(inputEmail);
            String password = getText(inputPassword);

            if (TextUtils.isEmpty(email) || TextUtils.isEmpty(password)) {
                Toast.makeText(this, "Nhap email va mat khau", Toast.LENGTH_SHORT).show();
                return;
            }

            btnLogin.setEnabled(false);
            auth.signInWithEmailAndPassword(email, password)
                    .addOnSuccessListener(result -> {
                        startActivity(new Intent(this, HomeActivity.class));
                        finish();
                    })
                    .addOnFailureListener(error -> {
                        btnLogin.setEnabled(true);
                        Toast.makeText(this, error.getMessage(), Toast.LENGTH_SHORT).show();
                    });
        });

        txtGoRegister.setOnClickListener(v ->
                startActivity(new Intent(this, RegisterActivity.class)));
    }

    private String getText(TextInputEditText input) {
        return input.getText().toString().trim();
    }
}
