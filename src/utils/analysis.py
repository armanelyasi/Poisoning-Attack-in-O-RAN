import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os

def create_or_load_dataframe(file_path, save_path, recreate=False):
    """
    Create or load a DataFrame. If 'recreate' is True, the DataFrame is recreated from the file.
    Otherwise, it is loaded from the saved path if available.

    :param file_path: Path to the CSV file.
    :param save_path: Path to save or load the transformed DataFrame.
    :param recreate: Whether to recreate the DataFrame or use the saved one.
    :return: Pandas DataFrame.
    """
    column_mapping = {
        "Name": "Attack",
        "State": "State",
        "Runtime": "Runtime",
        "charts/poisoning_rate": "Poisoning_Budget",
        "charts/AttackSuccessRate": "Attack_Success_Rate",
        "charts/episodic_length": "Episodic_Length",
        "charts/episodic_return": "Episodic_Return",
        "charts/reward_perturb_average": "Reward_Perturbation_Avg",
        "charts/reward_perturb_global": "Reward_Perturbation_Global",
        "global_step": "Global_Step_Count"
    }

    if not recreate and os.path.exists(save_path):
        try:
            print(f"Loading existing DataFrame from {save_path}...")
            return pd.read_csv(save_path)
        except Exception as e:
            print(f"Error loading saved DataFrame: {e}")
            return None

    try:
        print(f"Recreating DataFrame from {file_path}...")
        # Parse the CSV file
        df = pd.read_csv(file_path)
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Extract additional columns from 'Name'
        df['Kappa'] = df['Attack'].apply(
            lambda x: x.split('__')[-1] if x.startswith('Tr') else x.split('__')[-2] if x.startswith('SN') else None
        )
        df['Alpha'] = df['Attack'].apply(
            lambda x: x.split('__')[-1] if x.startswith('SN') else None
        )
        df['Attack'] = df['Attack'].apply(
            lambda x: 'TrojDRL' if x.startswith('Tr') else 'SleeperNets' if x.startswith('SN') else x
        )
        
        df= df[df["State"] == "finished"]
        
        # Save the DataFrame
        df.to_csv(save_path, index=False)
        print(f"Transformed DataFrame saved to {save_path}")
        return df
    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        return None

def replace_labels(label):
    return label.replace("_", " ").replace("Reward Perturbation Avg", "Average Reward Perturbation").replace("Attack Success Rate","Attack Success Rate (ASR)")

def plot_paretos(df, col_combinations, output_filenames):
    """
    Create 2D scatter plots for each pair of columns and save them as PDF files.
    
    :param df: Pandas DataFrame containing the data.
    :param col_combinations: List of tuples containing column pairs (x, y).
    :param output_filenames: List of filenames to save the plots.
    """
    try:
        for (x_col, y_col), output_filename in zip(col_combinations, output_filenames):
            plt.figure(figsize=(6, 4))
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue="Attack",
                style="Attack",  # Different markers for groups
                palette="husl",
                s=100,
                alpha=0.8,
                edgecolor=None  # Removes white border
            )
            x_rep = replace_labels(x_col)
            y_rep = replace_labels(y_col)
            plt.xlabel(x_rep)
            plt.ylabel(y_rep)
            plt.legend(title="Attack")
            plt.tight_layout(pad=0)  # Removes white borders around the figure
            plt.savefig(output_filename, bbox_inches='tight')  # Ensures no extra padding in output
            plt.close()
            print(f"2D scatter plot saved to {output_filename}")
    except Exception as e:
        print(f"Error generating 2D scatter plots: {e}")


def plot_3d_histogram_colored_by_attack(df, output_filename):
    """
    Create a 3D histogram showing the attack success rate grouped by Kappa, Budget, and Attack Type.
    The color of the bars is determined by the Attack type.

    :param df: Pandas DataFrame containing the data.
    :param output_filename: Filename to save the plot.
    """
    try:
        # Prepare grouped data for 3D plotting
        grouped = df.groupby(['Kappa', 'Poisoning_Budget', 'Attack'])['Attack_Success_Rate'].mean().reset_index()
        kappa = grouped['Kappa'].astype(str)
        budget = grouped['Poisoning_Budget']
        attack = grouped['Attack']
        success_rate = grouped['Attack_Success_Rate']

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert categorical variables to integers for plotting
        unique_kappas = {k: i for i, k in enumerate(sorted(kappa.unique()))}
        unique_attacks = {a: i for i, a in enumerate(sorted(attack.unique()))}
        kappa_numeric = kappa.map(unique_kappas)
        colors = sns.color_palette("husl", len(unique_attacks))
        color_map = {attack: colors[i] for i, attack in enumerate(unique_attacks)}

        # Plot 3D bars, colored by attack type
        for a in unique_attacks.keys():
            attack_filter = grouped['Attack'] == a
            ax.bar3d(
                kappa_numeric[attack_filter],
                budget[attack_filter],
                np.zeros_like(success_rate[attack_filter]),
                dx=0.4,
                dy=0.4,
                dz=success_rate[attack_filter],
                color=color_map[a],
                alpha=0.8,
                label=a
            )

        # Set labels and title
        ax.set_xticks(list(unique_kappas.values()))
        ax.set_xticklabels(unique_kappas.keys())
        ax.set_xlabel('Reward Perturbation ($\kappa$)')
        ax.set_ylabel(r'Poisoning Budget ($\beta$)')
        ax.set_zlabel('Average Attack Success Rate (ASR)')
        # ax.set_title('3D Histogram: Kappa, Budget (Y), and Attack Success Rate')

        # Add legend
        ax.legend(title="Attack")
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print(f"3D histogram saved to {output_filename}")
    except Exception as e:
        print(f"Error generating the 3D histogram: {e}")


def plot_asr_vs_beta(df, output_filename):
    """
    Generate a plot of ASR vs Beta with confidence intervals and connect points with lines for the same Attack and Victim.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the following columns:
        - 'Attack': Type of attack (determines marker style).
        - 'Victim': Victim identifier (determines color).
        - 'Beta': X-axis values.
        - 'ASR': Y-axis values.
        - 'ci': Confidence interval values for error bars.
    - output_filename (str): The filename to save the plot as a .pgf file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define color and marker mappings
    colors = {("TrojDRL", 1): 'blue',
               ("SleeperNets", 1): 'red',
               ("TrojDRL", 1): 'green',
               ("SleeperNets", 2): 'purple'
               }  
    

    # Group data by 'Attack' and 'Victim'
    grouped = df.groupby(['Attack', 'Victim'])

    for (attack, victim), group in grouped:
        # Sort by Beta for proper line plotting
        group = group.sort_values('Beta')
        
        # Plot lines connecting points for the same attack and victim
        ax.plot(
            group['Beta'],
            group['ASR'],
            linestyle='-',  # Solid line
            linewidth=1.5,
            color=colors.get((attack, victim), 'black'),
            label=f"{attack} (Victim set {victim})"
        )
        
        # Plot individual points with error bars
        ax.errorbar(
            group['Beta'],
            group['ASR'],
            yerr=group['ci'],
            # fmt=markers.get(victim, 'o'),  # Default marker is 'o'
            color=colors.get((attack, victim), 'black'),
            capsize=5,
            alpha=0.8
        )

    # Set axis labels and title
    ax.set_ylim([0, 1])
    ax.set_xlabel(r"Poisoning Budget ($\beta$)", fontsize=12)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_title("ASR vs Beta with Confidence Interval", fontsize=14)
    ax.legend(title="Attack (Victim)")
    # plt.grid(axis='x', color='0.95')
    plt.grid()

    # Save the plot as .pgf file
    try:
        fig.savefig(output_filename, format="pdf", bbox_inches='tight')
        print(f"Plot saved as '{output_filename}'")
    except Exception as e:
        print(f"Error saving plot as .pgf: {e}")

    plt.close(fig)


def plot_attack_heatmap(df, attack, output_filename):
    """
    Plots a heatmap of Attack Success Rate aggregated by Kappa and Poisoning Budget
    for a specific attack type and saves it to a file.

    Parameters:
    - df: pandas DataFrame containing the data.
    - attack: str, the name of the attack to filter by (column "Attack").
    - output_filename: str, the path to save the output heatmap PDF.
    """
    # Filter the dataframe by the specified attack
    df_filtered = df[df['Attack'] == attack]

    # Aggregate data for the heatmap
    df_pivot = df_filtered.pivot_table(
        index='Kappa',
        columns='Poisoning_Budget',
        values='Attack_Success_Rate',
        aggfunc='max'
    )

    # Plot heatmap
    font_size_labels = 28
    font_size_ticks = 22
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df_pivot, annot=False, cmap='bwr', cbar=False, square=False, linewidths=0.5)

    # Add colorbar manually to the left of the heatmap
    cbar = plt.colorbar(ax.collections[0], orientation="vertical")
    cbar.set_label("Attack Success Rate", fontsize=font_size_labels)  # Increase font size here

    plt.xlabel(r'Poisoning Budget, $\beta$', fontsize=font_size_labels)
    plt.ylabel('Reward Perturbation, $\kappa$', fontsize=font_size_labels, loc='center')
    # plt.title(f"Attack Success Rate for {attack}")

    # Rotate y-axis tick labels
    plt.yticks(rotation=0)  # You can change the angle to 90, 45, etc.

    # Adjust font size for tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=font_size_ticks)  # Increase x-axis tick label size
    ax.tick_params(axis='y', labelsize=font_size_ticks)  # Increase y-axis tick label size

    # Adjust colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size_ticks)  # Increase colorbar tick label size

    # Fine-tune layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust so that colorbar does not overlap with plot

    # Save to PDF
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_attack_scatter(df, attack, output_filename):
    """
    Plots a scatterplot of Attack Success Rate with Kappa and Poisoning Budget
    for a specific attack type and saves it to a file.

    Parameters:
    - df: pandas DataFrame containing the data.
    - attack: str, the name of the attack to filter by (column "Attack").
    - output_filename: str, the path to save the output scatterplot PDF.
    """
    # Filter the dataframe by the specified attack
    df_filtered = df[df['Attack'] == attack]

    # Plot scatterplot
    plt.figure(figsize=(6.4, 6.4))
    scatter = plt.scatter(
        df_filtered['Poisoning_Budget'],
        df_filtered['Kappa'],
        c=df_filtered['Attack_Success_Rate'],
        cmap='viridis',
        edgecolor='k',
        s=50
    )
    plt.colorbar(scatter, label='Attack Success Rate')
    plt.xlabel(r'Poisoning Budget ($\beta$)')
    plt.ylabel('Reward Perturbation ($\kappa$)')
    plt.title(f"Attack Success Rate Scatterplot for {attack}")

    # Save to PDF
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_online(df, feature, attack_time, output_filename):
    """
    Generate a plot of a selected feature over time from the results online dataframe

    Parameters:
    - df (pd.DataFrame): DataFrame containing the following columns:
        - 'time': y axis is time expressed in indication periodicities
        - 'dl_bytes': number of downlink bytes
        - 'dl_thp': downlink throughput
        - 'sl_prb': Number of assigned prbs
    - attack_time: 
    - output_filename (str): The filename to save the plot as a .pdf file.
    """
    label_map = {'dl_bytes': 'Downlink Buffer [kbyte]', 'dl_thp': 'Downlink Throughput [Mbps]','sl_prb': 'Allocated PRBs'}

    font_size_labels = 23
    font_size_ticks = 22
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if feature == 'dl_bytes':
        df_feature = df[feature].div(1000)
    else:
        df_feature = df[feature]

    df_time = df['time'] * 250 / 1000
    att_time = attack_time * 250 / 1000

    print(att_time)

    ax.plot(
            df_time,
            df_feature,
            linestyle='-',  # Solid line
            linewidth=1.5
        )

    ax.axvline(att_time, linestyle='--', linewidth=1.6, color='r') # set where the attack started
    ax.text(
        att_time, 
        df_feature.max() * 1.01,
        ' Attack Starts', 
        color='r', 
        fontsize=font_size_labels, 
        rotation=0, 
        verticalalignment='top', 
        horizontalalignment='left'
    )

    ax.tick_params(axis='x', labelsize=font_size_ticks)
    ax.tick_params(axis='y', labelsize=font_size_ticks) 
    ax.set_xlabel("Time [s]", fontsize=font_size_labels)
    ax.set_ylabel(label_map[feature], fontsize=font_size_labels, loc='top' if feature == 'dl_thp' else 'center')
    plt.grid()

    # Save the plot as .pgf file
    try:
        fig.savefig(output_filename, format="pdf", bbox_inches='tight')
        print(f"Plot saved as '{output_filename}'")
    except Exception as e:
        print(f"Error saving plot as .pgf: {e}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file and generate scatter plots.")
    parser.add_argument("--recreate", action="store_true", help="Recreate the DataFrame even if it exists.")
    args = parser.parse_args()

    # Set font sizes for all plots
    plt.rc('font', size=14)  # Base font size
    plt.rc('axes', titlesize=16, labelsize=18)  # Axis title and label sizes
    plt.rc('xtick', labelsize=12)  # X-axis tick labels
    plt.rc('ytick', labelsize=12)  # Y-axis tick labels
    plt.rc('legend', fontsize=12)  # Legend font size
    plt.rc('figure', titlesize=18)  # Figure title size

    # Hardcoded path to the CSV file
    file_path = "/datasets/recap.csv"
    save_path = "/datasets/transformed_recap.csv"

    # Create or load the DataFrame
    df = create_or_load_dataframe(file_path, save_path, recreate=args.recreate)

    # grouped_stats = df.groupby(['Attack', 'Poisoning_Budget'])['Attack_Success_Rate'].agg(['max', 'std'])
    # print(grouped_stats)

    # Plotting
    plot_paretos(
        df,
        [
            ("Reward_Perturbation_Avg", "Episodic_Return"),
            ("Reward_Perturbation_Avg", "Attack_Success_Rate"),
            ("Episodic_Return", "Attack_Success_Rate"),
        ],
        ["out/fig5/reward_vs_return.pdf", "out/fig5/reward_vs_success.pdf", "out/fig5/return_vs_success.pdf"],
    )

    # plot_attack_scatter(df, "TrojDRL", "out/fig6/trojdrl.pdf")
    # plot_attack_scatter(df, "SleeperNets", "out/fig6/sn.pdf")
 
    plot_attack_heatmap(df, "TrojDRL", "out/fig6/trojdrl.pdf")
    plot_attack_heatmap(df, "SleeperNets", "out/fig6/sn.pdf")

    df = pd.read_csv('/datasets/online.csv')
    plot_online(df, 'dl_bytes', 83, "out/online/dl_buffer_bytes.pdf")
    plot_online(df, 'dl_thp', 83, "out/online/dl_thp.pdf")
    plot_online(df, 'sl_prb', 83, "out/online/sl_prb.pdf")