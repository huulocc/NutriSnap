package com.huuloc.nutrisnap;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.android.material.button.MaterialButton;

import java.io.File;

public class ScanFoodActivity extends AppCompatActivity {

    private ImageView imagePreview;
    private Uri photoUri;
    private Uri selectedImageUri;

    // Launcher chụp ảnh từ Camera
    private final ActivityResultLauncher<Uri> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
                if (success && photoUri != null) {
                    selectedImageUri = photoUri;
                    imagePreview.setImageURI(selectedImageUri);
                    imagePreview.setVisibility(android.view.View.VISIBLE);
                }
            });

    // Launcher chọn ảnh từ Gallery
    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    selectedImageUri = uri;
                    imagePreview.setImageURI(selectedImageUri);
                    imagePreview.setVisibility(android.view.View.VISIBLE);
                }
            });

    // Launcher xin quyền Camera
    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    openCamera();
                } else {
                    Toast.makeText(this, "Cần cấp quyền Camera để chụp ảnh", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scan_food);

        imagePreview = findViewById(R.id.imagePreview);
        ImageView btnBack = findViewById(R.id.btnBack);
        MaterialButton btnCamera = findViewById(R.id.btnCamera);
        MaterialButton btnGallery = findViewById(R.id.btnGallery);
        MaterialButton btnAnalyze = findViewById(R.id.btnAnalyze);

        btnBack.setOnClickListener(v -> finish());

        btnCamera.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
            }
        });

        btnGallery.setOnClickListener(v -> galleryLauncher.launch("image/*"));

        btnAnalyze.setOnClickListener(v -> {
            if (selectedImageUri == null) {
                Toast.makeText(this, "Vui lòng chọn ảnh trước", Toast.LENGTH_SHORT).show();
                return;
            }
            Intent intent = new Intent(this, FoodResultActivity.class);
            intent.putExtra("imageUri", selectedImageUri.toString());
            startActivity(intent);
        });
    }

    private void openCamera() {
        File photoFile = new File(getExternalCacheDir(), "photo_" + System.currentTimeMillis() + ".jpg");
        photoUri = FileProvider.getUriForFile(this,
                getPackageName() + ".fileprovider", photoFile);
        cameraLauncher.launch(photoUri);
    }
}
