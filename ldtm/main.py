import argparse
import gc
import json

import torch

from .constants import INPUT_TRAIN_PATH, INPUT_VAL_PATH
from .generate import generate_song
from .songrnn import SongRNN
from .train import train
from .util import load_data, plot_losses

with open(INPUT_TRAIN_PATH, "r") as f:
    char_set = sorted(set(f.read()))

char_idx_map: dict[str, int] = {
    character: index for index, character in enumerate(char_set)
}
idx_char_map: dict[int, str] = {
    index: character for character, index in char_idx_map.items()
}

# TODO determine which device to use (cuda or cpu)
if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


def run(args):
    # Load the configuration from the specified config file
    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    # Extract configuration parameters
    MAX_GENERATION_LENGTH = config["max_generation_length"]
    TEMPERATURE = config["temperature"]
    SHOW_HEATMAP = config["show_heatmap"]
    generated_song_file_path = config["generated_song_file_path"]
    loss_plot_file_name = config["loss_plot_file_name"]
    evaluate_model_only = config["evaluate_model_only"]
    model_path = config["model_path"]

    # Load training and validation data
    data = load_data(INPUT_TRAIN_PATH, config)
    data_val = load_data(INPUT_VAL_PATH, config)

    print("==> Building model..")

    in_size, out_size = len(char_set), len(char_set)
    # Initialize the SongRNN model
    model = SongRNN(in_size, out_size, config)

    # If evaluating model only and trained model path is provided:
    if evaluate_model_only and model_path != "":
        # Load the checkpoint from the specified model path
        model = torch.load(model_path, map_location=device)
        print(f"==> Model loaded from checkpoint to {device}..")
    else:
        # Train the model and get the training and validation losses
        losses, v_losses = train(model, data, data_val, char_idx_map, config, device)

        # Plot the training and validation losses
        plot_losses(losses, v_losses, loss_plot_file_name)

    # As a fun exercise, after your model is well-trained you can see how the model extends Beethoven's famous fur-elise tune
    # with open("./data/fur_elise.txt", 'r') as file:
    # prime_str = file.read()
    # print("Prime str = ", prime_str)

    if args.fur_elise:
        with open("./data/test_fe.txt", "r") as file:
            prime_str = file.read()
    else:
        prime_str = "<start>"

    print("Prime str = ", prime_str)
    # Generate a song using the trained model
    generated_song = generate_song(
        model,
        device,
        char_idx_map,
        idx_char_map,
        max_len=MAX_GENERATION_LENGTH,
        temp=TEMPERATURE,
        prime_str=prime_str,
        show_heatmap=SHOW_HEATMAP,
    )

    # Write the generated song to a file
    with open(generated_song_file_path, "w") as file:
        file.write(generated_song)

    print("Generated song is written to : ", generated_song_file_path)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.json", help="Specify the config file"
    )
    parser.add_argument("--fur-elise", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
