import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Required for creating DataFrame for seaborn


class Bandit:
    # Default class-level mean and variance
    mean = 10
    variance = 3

    def __init__(self, mean=None, variance=None):
        # Use provided mean/variance or fall back to class defaults
        self.mean = mean if mean is not None else Bandit.mean
        self.variance = variance if variance is not None else Bandit.variance
        # Offset is a random value specific to each bandit instance
        self.offset = random.uniform(-3, 3)

    def pull(self):
        """
        Simulates pulling the bandit arm and returns a reward.
        The reward is sampled from a normal distribution.
        """
        return np.random.normal(self.mean + self.offset, np.sqrt(self.variance))

    def plot(self, num_samples=5000, bandit_label=""):
        """
        Generates samples by repeatedly pulling the bandit and then plots
        the distribution of these sampled rewards using a violin plot.

        Args:
            num_samples (int): The number of samples to pull from the bandit
                                to estimate its distribution for the plot.
            bandit_label (str): An optional label for the bandit on the x-axis.
        """
        print(f"Generating {num_samples} pulls for plotting...")
        pulls = [self.pull() for _ in range(num_samples)]

        # Create a DataFrame for seaborn plotting.
        # This is the 'sampling' step to get data points.
        df = pd.DataFrame({'Reward': pulls, 'Bandit': bandit_label if bandit_label else 'Single Bandit'})

        plt.figure(figsize=(7, 5))
        # Create the violin plot from the sampled data.
        # 'inner="quartile"' shows the median and quartiles of the sampled data.
        sns.violinplot(x='Bandit', y='Reward', data=df, color='skyblue', inner='quartile',
                       saturation=0.7, linewidth=1.5, cut=0)

        # Calculate the true expected value (mean) of this specific bandit's underlying distribution
        true_expected_value = self.mean + self.offset

        # Add a horizontal line to indicate the true expected value for comparison
        # The x-coordinate for the line is 0 because there's only one 'Bandit' category on the x-axis
        plt.hlines(true_expected_value, -0.2, 0.2, color='red', linestyle='--', linewidth=2,
                   label=f'True Mean: {true_expected_value:.2f}')

        plt.title(f'Sampled Reward Distribution for Bandit (Offset: {self.offset:.2f})')
        plt.ylabel('Reward Value')
        plt.xlabel('')  # Remove x-axis label as 'Bandit' category is clear
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('bandit_sampled_reward_distribution.png')  # Save the plot
        plt.close()  # Close the plot to free memory
        print("Plot saved as 'bandit_sampled_reward_distribution.png'")


# --- Example Usage ---
if __name__ == "__main__":
    # Create an instance of the Bandit class using default parameters
    my_bandit = Bandit()
    print(
        f"Bandit created with base mean: {my_bandit.mean}, variance: {my_bandit.variance}, and offset: {my_bandit.offset:.2f}")

    # Plot the distribution of this bandit's rewards by sampling
    my_bandit.plot(num_samples=10000, bandit_label="My Bandit")

    # Create another bandit with custom parameters to see a different distribution
    custom_bandit = Bandit(mean=5, variance=2)
    print(
        f"\nCustom Bandit created with base mean: {custom_bandit.mean}, variance: {custom_bandit.variance}, and offset: {custom_bandit.offset:.2f}")
    custom_bandit.plot(num_samples=8000, bandit_label="Custom Bandit")