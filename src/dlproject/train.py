import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, loader, loss_fn, device):
    '''Returns (average loss, accuracy) of `model` on the data in `loader`.'''
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-3, backbone_lr=None, checkpoint_path=None):
    if backbone_lr is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam([
            {"params": model.layer4.parameters(), "lr": backbone_lr},
            {"params": [p for n, p in model.named_parameters() if "layer4" not in n], "lr": lr}
        ])
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss, train_acc = running_loss / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            marker = " *"
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.3f}{marker}")

    print(f"Best: epoch {best_epoch}, val loss {best_val_loss:.4f}  (* = checkpoint saved)")
    history["best_epoch"] = best_epoch
    return history
