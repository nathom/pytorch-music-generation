import os
import sys

import torch
import torch.nn as nn

from . import util
from .songrnn import SongRNN


def train(
    model: SongRNN,
    data: list[str],
    data_val: list[str],
    char_idx_map: dict[str, int],
    config,
    device: torch.device,
):
    """
    Train the provided model using the specified configuration and data.

    Parameters:
    ----------
    - model (nn.Module): The neural network model to be trained
    - data (list): A list of training data sequences
    - data_val (list): A list of validation data sequences
    - char_idx_map (dict): A dictionary mapping characters to their corresponding indices
    - config (dict): A dictionary containing configuration parameters for training:
    - device (torch.device): The device (e.g., "cpu" or "cuda") on which the model is located

    Returns:
    -------
    - losses (list): A list containing training losses for each epoch
    - v_losses (list): A list containing validation losses for each epoch
    """

    # Extracting configuration parameters
    N_EPOCHS = config["no_epochs"]
    LR = config["learning_rate"]
    SAVE_EVERY = config["save_epoch"]
    MODEL_TYPE = config["model_type"]
    HIDDEN_SIZE = config["hidden_size"]
    DROPOUT_P = config["dropout"]
    SEQ_SIZE = config["sequence_size"]
    CHECKPOINT = "ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}".format(
        MODEL_TYPE, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []

    # Training over epochs
    hidden = model.init_hidden(1, device)  # Zero out the hidden layer
    for epoch in range(N_EPOCHS):
        # TRAIN: Train model over training data
        total_loss = 0.0
        for i in range(len(data)):
            optimizer.zero_grad()  # Zero out the gradient
            hidden = model.detach_hidden(hidden)

            # Get random sequence from data
            input_seq, target_seq = util.get_random_song_sequence_target(
                data[i], char_idx_map, SEQ_SIZE
            )
            input_seq = input_seq.unsqueeze(0)
            target_seq = target_seq.unsqueeze(0)
            # Move to device
            input_seq = torch.tensor(input_seq, dtype=torch.long).to(device)
            target_seq = torch.tensor(target_seq, dtype=torch.long).to(device)

            output: torch.Tensor

            output, hidden = model(input_seq, hidden)

            # reshape data
            output = output.view(-1, len(char_idx_map))
            target_seq = target_seq.view(-1)
            loss = criterion(output, target_seq)

            loss.backward()
            # TODO: clip grad norm?
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            avg_loss_per_sequence = loss.item()
            total_loss += avg_loss_per_sequence

            # Display progress
            msg = "\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}".format(
                epoch, (i + 1) / len(data) * 100, i, avg_loss_per_sequence
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        print()

        # Append the avg loss on the training dataset to train_losses list

        train_losses.append(total_loss / len(data))

        # VAL: Evaluate Model on Validation dataset
        model.eval()  # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
            val_loss_total = 0.0
            for i in range(len(data_val)):
                hidden = model.init_hidden(1, device)  # Zero out the hidden layer

                input_seq, target_seq = util.get_random_song_sequence_target(
                    data_val[i], char_idx_map, SEQ_SIZE
                )
                # add batch size dim
                input_seq = input_seq.unsqueeze(0)
                target_seq = target_seq.unsqueeze(0)

                input_seq = torch.tensor(input_seq, dtype=torch.long).to(device)
                target_seq = torch.tensor(target_seq, dtype=torch.long).to(device)

                output, _ = model(input_seq, hidden)

                val_loss = criterion(
                    output.view(-1, len(char_idx_map)), target_seq.view(-1)
                )

                avg_loss_per_sequence = val_loss.item()
                val_loss_total += avg_loss_per_sequence

                # Display progress
                msg = "\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}".format(
                    epoch, (i + 1) / len(data_val) * 100, i, avg_loss_per_sequence
                )
                sys.stdout.write(msg)
                sys.stdout.flush()

            print()

        # Append the avg loss on the validation dataset to validation_losses list
        validation_losses.append(val_loss_total / len(data_val))

        model.train()  # TURNING THE TRAIN MODE BACK ON !

        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        # Save checkpoint.
        if (epoch % SAVE_EVERY == 0 and epoch != 0) or epoch == N_EPOCHS - 1:
            print("=======>Saving..")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                "./checkpoint/" + CHECKPOINT + ".t%s" % epoch,
            )

    return train_losses, validation_losses
