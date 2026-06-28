package com.huuloc.nutrisnap;

import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

public class RegisterActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        MaterialButton btnRegister = findViewById(R.id.btnRegister);
        TextView txtGoLogin = findViewById(R.id.txtGoLogin);

        btnRegister.setOnClickListener(v -> finish());
        txtGoLogin.setOnClickListener(v -> finish());
    }
}
