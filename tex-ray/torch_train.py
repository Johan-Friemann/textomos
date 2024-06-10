import torch


def one_epoch(model, criterion, optimizer, dataloader, device, mode):
    """DOCSTRING"""
    if mode == "train":
        model.train()
    else:
        model.eval()

    num_batches = len(dataloader)
    epoch_loss = 0.0
    current_batch = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device).unsqueeze(1)  # Unsqueeze channel axis
        labels = labels.to(device)
        current_batch += 1

        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == "train"):
            outputs = model(inputs)["out"]
            loss = criterion(outputs, labels)
            if mode == "train":
                loss.backward()
                optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        print(
            "|| Epoch progress: ",
            "{:6.2%}".format(current_batch / num_batches),
            "| Current batch loss: {:.6f} ||".format(batch_loss),
            end="\r",
        )

    print(" " * 61, end="\r") # Clear such that following text prints cleanly.
    return epoch_loss / num_batches
