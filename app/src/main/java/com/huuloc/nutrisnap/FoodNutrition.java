package com.huuloc.nutrisnap;

import java.util.HashMap;
import java.util.Map;

public class FoodNutrition {

    public static class Info {
        public final int calories;
        public final int protein;

        Info(int calories, int protein) {
            this.calories = calories;
            this.protein = protein;
        }
    }

    private static final Map<String, Info> TABLE = new HashMap<>();

    static {
        TABLE.put("pho_bo", new Info(420, 25));
        TABLE.put("banh_mi", new Info(350, 14));
        TABLE.put("com_tam", new Info(600, 30));
        TABLE.put("bun_bo", new Info(480, 28));
        TABLE.put("goi_cuon", new Info(180, 11));
        TABLE.put("banh_xeo", new Info(500, 18));
        TABLE.put("mi_quang", new Info(450, 22));
        TABLE.put("xoi", new Info(400, 8));
        TABLE.put("chao", new Info(250, 10));
        TABLE.put("com_ga", new Info(550, 32));
        TABLE.put("bun_thit_nuong", new Info(470, 24));
        TABLE.put("bun_rieu", new Info(400, 18));
        TABLE.put("hu_tieu", new Info(430, 20));
        TABLE.put("banh_cuon", new Info(300, 12));
        TABLE.put("cha_gio", new Info(350, 15));
    }

    public static Info get(String modelLabel) {
        Info info = TABLE.get(modelLabel);
        return info != null ? info : new Info(400, 20);
    }
}
