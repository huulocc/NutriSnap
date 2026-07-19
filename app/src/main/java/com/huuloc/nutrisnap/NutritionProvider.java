package com.huuloc.nutrisnap;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.android.gms.tasks.Tasks;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;

public class NutritionProvider extends ContentProvider {

    public static final String AUTHORITY = "com.huuloc.nutrisnap.provider";
    public static final Uri FOODS_URI = Uri.parse("content://" + AUTHORITY + "/foods");

    public static final String COL_DISPLAY_NAME = "display_name";
    public static final String COL_CALORIES = "calories";
    public static final String COL_PROTEIN = "protein";

    @Override
    public boolean onCreate() {
        return true;
    }

    @Nullable
    @Override
    public Cursor query(@NonNull Uri uri, @Nullable String[] projection, @Nullable String selection,
                        @Nullable String[] selectionArgs, @Nullable String sortOrder) {
        String modelLabel = uri.getLastPathSegment();
        MatrixCursor cursor = new MatrixCursor(
                new String[]{COL_DISPLAY_NAME, COL_CALORIES, COL_PROTEIN});

        if (modelLabel == null) {
            return cursor;
        }

        try {
            DocumentSnapshot doc = Tasks.await(FirebaseFirestore.getInstance()
                    .collection("foods").document(modelLabel).get());
            if (doc.exists()) {
                String name = doc.getString("displayName");
                Long calories = doc.getLong("defaultCalories");
                Long protein = doc.getLong("defaultProtein");
                cursor.addRow(new Object[]{
                        name,
                        calories == null ? 0L : calories,
                        protein == null ? 0L : protein
                });
            }
        } catch (Exception ignored) {
        }
        return cursor;
    }

    @Nullable
    @Override
    public String getType(@NonNull Uri uri) {
        return "vnd.android.cursor.item/food";
    }

    @Nullable
    @Override
    public Uri insert(@NonNull Uri uri, @Nullable ContentValues values) {
        return null;
    }

    @Override
    public int delete(@NonNull Uri uri, @Nullable String selection, @Nullable String[] selectionArgs) {
        return 0;
    }

    @Override
    public int update(@NonNull Uri uri, @Nullable ContentValues values, @Nullable String selection,
                      @Nullable String[] selectionArgs) {
        return 0;
    }
}
