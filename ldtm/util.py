import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(file, config):
    """
    Load song data from a file.

    Parameters:
    ----------
    - file (str): The path to the data file
    - config (dict): A dictionary containing configuration parameters:
        - "sequence_size" (int): The size of the sequences to extract from the data and train our model.
        (We want to ensure that the song we use is atleast as big as the sequence size)

    Returns:
    -------
    - data (list): A list of sequences extracted from the data file
    """

    # Extract configuration parameters
    SEQ_SIZE = config["sequence_size"]

    # Initialize an empty list to store the data
    data = []

    # Read data from the file
    with open(file, "r") as f:
        buffer = ""
        for line in f:
            if line == "<start>\n":
                buffer = line
            elif line == "<end>\n":
                buffer += line
                # Check if the buffer size meets the sequence size threshold
                if len(buffer) > SEQ_SIZE + 10:
                    data.append(buffer)
                buffer = ""
            else:
                buffer += line

    return data


def get_random_sequence(data, sequence_length):
    """
    TODO: Retrieves a random slice of the given data with the specified sequence length.

    Args:
    ----
        data (str): The input data (e.g., song notation).
        sequence_length (int): The desired length of the sequence to extract.

    Returns:
    -------
        list: A random slice of the input data with the specified sequence length.
    """
    start_index = random.randint(0, len(data) - sequence_length - 1)
    return data[start_index : start_index + sequence_length]


def characters_to_tensor(sequence, char_idx_map):
    """
    Converts a sequence of characters to a PyTorch tensor using the provided character set.
    (DON'T CHANGE)

    Args:
    ----
        sequence (str): The sequence of characters to convert.
        char_idx_map (dict): A map of characters to their index

    Returns:
    -------
        torch.Tensor: A PyTorch tensor representing the input sequence.
    """
    return torch.tensor(
        [char_idx_map[character] for character in sequence], dtype=torch.long
    )


def get_random_song_sequence_target(song, char_idx_map, sequence_length, device):
    """
    Retrieves a random sequence from the given song data along with its target sequence.
    (DON'T CHANGE)

    Args:
    ----
        song (str): The song data, represented as a string.
        char_idx_map (dict): A map of characters to their index
        sequence_length (int): The desired length of the sequence to extract.

    Returns:
    -------
        tuple: A tuple containing the PyTorch tensor representing the input sequence
               and the PyTorch tensor representing the target sequence.
    """
    sequence = get_random_sequence(song, sequence_length)
    sequence_tensor = characters_to_tensor(sequence[:-1], char_idx_map)
    target_tensor = characters_to_tensor(sequence[1:], char_idx_map)
    sequence_tensor, target_tensor = (
        sequence_tensor.unsqueeze(0).to(device),
        target_tensor.unsqueeze(0).to(device),
    )
    return sequence_tensor, target_tensor


def get_character_from_index(char_idx_map, index):
    """
    (DON'T CHANGE)
    """
    for character, idx in char_idx_map.items():
        if idx == index:
            return character
    # If value is not found, return None or raise an error as needed
    return None


def plot_losses(train_losses, val_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument).

    Args:
    ----
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
    -------
        None
    """

    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir("./writeup/plots"):
        os.mkdir("./writeup/plots")

    # Plotting training and validation losses
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./writeup/plots/" + fname + ".png")


def pad(song, data, pad_factor=20):
    """
    Utility function to pad the generated song and the corresponding heatmap values.
    (DON'T CHANGE)

    Args:
    ----
        song (list): generated song.
        data (list): heatmap values.
        pad_factor (itn): padding data to a multiple of pad_factor.

    Returns:
    -------
        padded sequences
    """
    padded_song = np.asarray(
        list(song.ljust((len(song) // pad_factor + 1) * pad_factor))
    )
    padded_data = np.pad(
        data,
        (0, ((pad_factor - ((len(data)) % pad_factor)) % pad_factor)),
        mode="constant",
        constant_values=0.0,
    )
    return padded_song, padded_data


def show_values(pc, song, fmt="%.2f", **kw):
    """
    Utility function to plot the heatmap.
    (DON'T CHANGE)

    Returns:
    -------
        None
    """
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)

        x_idx, y_idx = int(x - 0.5), int(y - 0.5)

        if y_idx < len(song) and x_idx < len(song[0]):
            ax.text(
                x,
                y,
                repr(song[y_idx][x_idx])[1:-1],
                fontsize=45,
                ha="center",
                va="center",
                color=color,
                **kw,
            )
