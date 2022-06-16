import torch.nn as nn
import torch
# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # early stop trigger
    patience = 50
    trigger = 0

    # history
    loss_list = []
    acc_list = []
    prev_acc = -1

    # Repeat for each epoch
    for epoch in range(1,num_epochs+1):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction

        if epoch > 1:
            if acc <= prev_acc:
                trigger += 1
            else:
                prev_acc = acc
                trigger = 0

        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Trigger: {trigger}')
        loss_list.append(loss.item())
        acc_list.append(acc)
        

        if trigger >= patience:
            break

        if epoch % 10 == 0:
            torch.save(model, "model_" + str(epoch) + ".pth")

    print('Finished Training')
    return loss_list, acc_list

