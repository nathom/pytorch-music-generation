import matplotlib.pyplot as plt
import numpy as np
import torch

from .SongRNN import SongRNN
from .util import characters_to_tensor, pad, show_values


def generate_song(
    model,
    device,
    char_idx_map,
    idx_char_map,
    max_len=1000,
    temp=0.8,
    prime_str="<start>",
    show_heatmap=False,
):
    """
    Generates a song using the provided model.

    Parameters:
    ----------
    - model (songrnn.SongRNN): The trained model used for generating the song
    - device (torch.device): The device (e.g., "cpu" or "cuda") on which the model is located
    - char_idx_map (dict): A map of characters to their index
    - max_len (int): The maximum length of the generated song
    - temp (float): Temperature parameter for temperature scaling during sampling
    - prime_str (str): Initialize the beginning of the song
    - show_heatmap (bool): Flag to show the heatmap (if implemented)

    Returns:
    -------
    - generated_song (str): The generated song as a string
    """

    # Move model to the specified device and set the model to evaluation mode
    model: SongRNN = model.to(device)
    model.eval()

    # Initialize the hidden state
    hidden = model.init_hidden(1, device)

    all_hidden_list = []
    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        # "build up" hidden state using the beginning of a song '<start>'
        generated_song = prime_str
        prime = characters_to_tensor(generated_song, char_idx_map)
        for i in range(len(prime)):
            all_hidden_list.append(hidden)
            c = prime[i].unsqueeze(0).unsqueeze(0).to(device)
            _, hidden = model(c, hidden)

        # Continue generating the rest of the sequence until reaching the maximum length or encountering the end token.
        for _ in range(max_len - len(prime_str)):
            all_hidden_list.append(hidden)

            input_char = (
                characters_to_tensor(generated_song[-1], char_idx_map)
                .unsqueeze(0)
                .to(device)
            )
            output, hidden = model(input_char, hidden)

            out = np.array(np.exp(output.cpu().numpy() / temp))
            dist = out / np.sum(out)
            out = out.squeeze()
            dist = dist.squeeze()

            next_char = idx_char_map[np.random.choice(len(dist), p=dist)]

            # Add the generated character to the `generated_song`
            generated_song += next_char

            # If the generated character is an end token, break the loop
            if next_char == "<end>":
                break

    # Turn the model back to training mode
    model.train()

    if show_heatmap:
        heatmap = torch.FloatTensor([all_hidden_list[i][0][0][0].cpu().numpy() for i in range(len(all_hidden_list))])
        print(f"Heatmap original shape: {heatmap.shape}")
        generate_heatmap(generated_song, heatmap, 10)

    return generated_song


def sample_from_distribution(distribution):
    # Sample a character index from a probability distribution
    return torch.multinomial(distribution, 1).item()


def generate_heatmap(generated_song, heatmap, neuron_idx=0):
    """
    Generates a heatmap using the provided generated song, heatmap chart values and neuron id.

    Parameters:
    ----------
    - generated_song (nn.Module): The song generated by a trained model.
    - heatmap (torch.Tensor): heatmap/activation values from a particular layer of the trained model.
    - neuron_idx (int): id of the neuron to plot heatmap for.

    Returns:
    -------
        None
    """
    pad_factor = 20
    heatmap = heatmap.detach().numpy()

    data = np.append(heatmap[:, neuron_idx], 0.0)
    padded_song, padded_data = pad(generated_song, data, pad_factor=pad_factor)

    padded_song = np.reshape(padded_song, (len(padded_song) // pad_factor, pad_factor))
    padded_data = np.reshape(padded_data, (len(padded_data) // pad_factor, pad_factor))

    plt.figure(figsize=(heatmap.shape[0] // 4, heatmap.shape[1] // 4))
    plt.title(f"Heatmap For Song RNN, Neuron ID: {neuron_idx}")
    heatplot = plt.pcolor(
        padded_data, edgecolors="k", linewidths=4, cmap="RdBu_r", vmin=-1.0, vmax=1.0
    )

    show_values(heatplot, song=padded_song)
    plt.colorbar(heatplot)
    plt.gca().invert_yaxis()
    plt.savefig(f"./writeup/plots/heatmap_{neuron_idx}.png")
    print(f"==> Heatmap saved for Neuron ID: {neuron_idx}..")
    return