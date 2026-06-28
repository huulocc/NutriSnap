package com.huuloc.nutrisnap;

public class Meal {
    public String name;
    public String portion;
    public String calories;
    public String protein;

    public Meal(String name, String portion, String calories, String protein) {
        this.name = name;
        this.portion = portion;
        this.calories = calories;
        this.protein = protein;
    }
}
