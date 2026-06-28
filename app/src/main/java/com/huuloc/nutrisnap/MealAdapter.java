package com.huuloc.nutrisnap;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class MealAdapter extends RecyclerView.Adapter<MealAdapter.MealHolder> {

    private final List<Meal> meals;

    public MealAdapter(List<Meal> meals) {
        this.meals = meals;
    }

    @NonNull
    @Override
    public MealHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_meal, parent, false);
        return new MealHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MealHolder holder, int position) {
        Meal meal = meals.get(position);
        holder.name.setText(meal.name);
        holder.portion.setText(meal.portion);
        holder.calories.setText(meal.calories);
        holder.protein.setText(meal.protein);
    }

    @Override
    public int getItemCount() {
        return meals.size();
    }

    static class MealHolder extends RecyclerView.ViewHolder {
        TextView name;
        TextView portion;
        TextView calories;
        TextView protein;

        MealHolder(@NonNull View itemView) {
            super(itemView);
            name = itemView.findViewById(R.id.txtMealName);
            portion = itemView.findViewById(R.id.txtMealPortion);
            calories = itemView.findViewById(R.id.txtMealCalories);
            protein = itemView.findViewById(R.id.txtMealProtein);
        }
    }
}
