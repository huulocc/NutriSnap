package com.huuloc.nutrisnap;

import android.Manifest;
import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.IBinder;
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
    private MaterialButton btnAnalyze;
    private Uri photoUri;
    private Uri selectedImageUri;

    private FoodRecognitionService recognitionService;
    private boolean serviceBound = false;

    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            recognitionService = ((FoodRecognitionService.LocalBinder) binder).getService();
            serviceBound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            serviceBound = false;
        }
    };

    private final ActivityResultLauncher<Uri> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
                if (success && photoUri != null) {
                    selectedImageUri = photoUri;
                    imagePreview.setImageURI(selectedImageUri);
                    imagePreview.setVisibility(android.view.View.VISIBLE);
                }
            });

    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    selectedImageUri = uri;
                    imagePreview.setImageURI(selectedImageUri);
                    imagePreview.setVisibility(android.view.View.VISIBLE);
                }
            });

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
        btnAnalyze = findViewById(R.id.btnAnalyze);

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
            analyzeImage();
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
        bindService(new Intent(this, FoodRecognitionService.class), connection, BIND_AUTO_CREATE);
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (serviceBound) {
            unbindService(connection);
            serviceBound = false;
        }
    }

    private void analyzeImage() {
        if (!serviceBound) {
            Toast.makeText(this, "Dịch vụ nhận diện chưa sẵn sàng", Toast.LENGTH_SHORT).show();
            return;
        }

        btnAnalyze.setEnabled(false);
        btnAnalyze.setText("Đang nhận diện...");

        recognitionService.recognize(selectedImageUri, new PredictionApi.Callback() {
            @Override
            public void onSuccess(String modelLabel, String displayName, double confidence) {
                Intent intent = new Intent(ScanFoodActivity.this, FoodResultActivity.class);
                intent.putExtra("imageUri", selectedImageUri.toString());
                intent.putExtra("modelLabel", modelLabel);
                intent.putExtra("displayName", displayName);
                intent.putExtra("confidence", confidence);
                startActivity(intent);

                btnAnalyze.setEnabled(true);
                btnAnalyze.setText(R.string.analyze);
            }

            @Override
            public void onError(String message) {
                Toast.makeText(ScanFoodActivity.this, message, Toast.LENGTH_LONG).show();
                btnAnalyze.setEnabled(true);
                btnAnalyze.setText(R.string.analyze);
            }
        });
    }

    private void openCamera() {
        File photoFile = new File(getExternalCacheDir(), "photo_" + System.currentTimeMillis() + ".jpg");
        photoUri = FileProvider.getUriForFile(this,
                getPackageName() + ".fileprovider", photoFile);
        cameraLauncher.launch(photoUri);
    }
}
