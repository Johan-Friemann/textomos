import torch


def one_epoch(model, criterion, optimizer, dataloader, device, mode):
    """Process one (1) epoch. This can be either training or validation.


    Args:
        model (torch model): The model to train/validate.

        criterion (torch loss function): The loss function to use for training/
                                         validation.

        optimizer (torch optimizer): Optimizer to use for training.

        dataloader (torch dataloader): Dataloader for loading data for training/
                                       validation.

        device (torch device): The device (cpu/gpu) that will be used to load/
                               train the model.

        mode (str): If set to "training" the model will be trained, else the
                    model is validated.

    Keyword args:
        -

    Returns:
        epoch_loss (float): The epoch loss. This is the sum of all batch losses
                            divided by the number of batches in the epoch.
    """
    if mode == "training":
        header = "Training"
        model.train()
    else:
        header = "Evaluation"
        model.eval()

    num_batches = len(dataloader)
    epoch_loss = 0.0
    current_batch = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device).unsqueeze(1)  # Unsqueeze channel axis
        labels = labels.to(device)
        current_batch += 1

        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == "training"):
            outputs = model(inputs)["out"]
            loss = criterion(outputs, labels)
            if mode == "training":
                loss.backward()
                optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        loss_str = "{:.6f}".format(batch_loss)
        progress_str = "{:7.2%}".format(current_batch / num_batches)

        num_space = 28 - len(header) - len(loss_str) - len(progress_str)
        print(
            "||" + " " * (num_space // 2 - 1) + header + " progress: ",
            progress_str + " | Current batch loss: " + loss_str,
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
            end="\r",
        )

    print(" " * 67, end="\r")  # Clear such that following text prints cleanly.
    return epoch_loss / num_batches


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    device,
    num_epochs,
    state_dict_path="./tex-ray/state_dict.pt",
):
    """Train a torch model for one or more epochs.


    Args:
        model (torch model): The model to train.

        criterion (torch loss function): The loss function to use for training.

        optimizer (torch optimizer): Optimizer to use for training.

        dataloaders (dict(torch dataloader)): A dictionary containing two items.
                                              The first key "training" refers to
                                              the dataloader to use for training
                                              while the second key "validation"
                                              refers to the dataloader to use
                                              for validation.

        device (torch device): The device (cpu/gpu) that will be used to load/
                               train the model.

        num_epochs (int): The number of epochs to train the model.

    Keyword args:
        state_dict_path (str): The absolute path to where to save the state
                               dictionary. The function continuously saves the
                               set of weights that results in the current lowest
                               validation loss

    Returns:
        -
    """
    num_space = 32 - len(model.__class__.__name__) - len(str(num_epochs))

    print(
        "-" * 66 + "\n||" + " " * (num_space // 2 - 1),
        f"Training model: '{model.__class__.__name__}' for {num_epochs} epochs",
        " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        "\n" + "-" * 66,
    )

    best_loss = (
        10e6  # Just put a big value to ensure non-convergence at step 1.
    )
    for epoch in range(num_epochs):
        num_space = 55 - len(str(epoch + 1)) - len(str(num_epochs))
        print(
            "||" + " " * (num_space // 2 - 1),
            f"Epoch {epoch + 1}/{num_epochs}",
            " " * (num_space // 2 + (num_space % 2) - 1) + "||",
        )

        for mode in ["training", "validation"]:
            epoch_loss = one_epoch(
                model, criterion, optimizer, dataloaders[mode], device, mode
            )
            loss_str = "{:.6f}".format(epoch_loss)
            if mode == "training":
                num_space = 41 - len(loss_str)
                print(
                    "||" + " " * (num_space // 2 - 1),
                    "Training epoch loss: " + loss_str,
                    " " * (num_space // 2 + (num_space % 2) - 1) + "||",
                )
            if mode == "validation":
                num_space = 39 - len(loss_str)
                print(
                    "||" + " " * (num_space // 2 - 1),
                    "Validation epoch loss: " + loss_str,
                    " " * (num_space // 2 + (num_space % 2) - 1) + "||",
                )
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), state_dict_path)
        print("-" * 66)

    return None
