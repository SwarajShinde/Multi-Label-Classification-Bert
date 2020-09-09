def run():

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} /{epochs}")
        print('-' * 10)
        

        train_acc,train_loss = train_fn(train_dataloader,model,optimizer,loss_fn,DEVICE,scheduler,len(df_train))
        
        print(f"Training Losses {train_loss} accuracy {train_acc}")
        

        eval_acc,eval_loss = eval_fn(valid_dataloader,model,DEVICE,loss_fn,len(df_valid))

        print(f'Val loss {eval_loss} accuracy {eval_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['eval_acc'].append(eval_acc)
        history['eval_loss'].append(eval_loss)

        if eval_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = eval_acc



# Inference Loop

axes = test_fn(model,test_dataloader,DEVICE)
