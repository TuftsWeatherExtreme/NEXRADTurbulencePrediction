# plot_epochs.py
# Team Celestial Blue
# Spring 2025
# Last Modified: 05/06/2025
# This file exports a function that can be used to produce epoch plots
# to determine if the model is overfitting

import os
import matplotlib as plt

def plot_epochs(output_dir, formatted_curr_date, NUM_EPOCHS, l2_alpha_list,
                 training_losses_by_l2, validation_losses_by_l2):
    
    
    os.makedirs(output_dir, exist_ok=True)

    for l2 in l2_alpha_list:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NUM_EPOCHS + 1), training_losses_by_l2[l2], label="Training Loss")
        plt.plot(range(1, NUM_EPOCHS + 1), validation_losses_by_l2[l2], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epoch (L2 = {l2})")
        plt.legend()
        plt.grid(True)

        # Save the figure
        l2_clean = str(l2).replace(".", "_")
        fig_path = os.path.join(output_dir, f"{formatted_curr_date}_l2_{l2_clean}_loss_plot.png")
        plt.savefig(fig_path)
        print(f"Saved plot for L2={l2} to: {fig_path}")
        plt.close()