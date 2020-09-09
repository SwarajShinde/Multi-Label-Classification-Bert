class Engine():
  def train_fn(data_loader,model,optimizer,loss_fn,device,scheduler,n_examples):
      model = model.train()
      losses = []
      correct_predictions = 0
      tkt = tqdm(data_loader,total = len(data_loader))
      for d in tkt:
          ids = d['ids'].to(device)
          mask = d['mask'].to(device)
          targets = d['target'].to(device)

          outputs = model(ids=ids,mask=mask)
          _, preds = torch.max(outputs, dim=1)
          loss = loss_fn(outputs,targets)

          correct_predictions += torch.sum(preds==targets)
          losses.append(loss.item())

          loss.backward()
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()

      #print(f"Individual Training Losses {loss.item()} " )
      return correct_predictions.double() / n_examples, np.mean(losses)


  def eval_fn(data_loader,model,device,loss_fn,n_examples):
      model = model.eval()
      losses= []
      correct_predictions = 0
      with torch.no_grad():
          tke = tqdm(data_loader,total = len(data_loader))
          for d in tke:
              ids = d['ids'].to(device)
              mask = d['mask'].to(device)
              targets = d['target'].to(device)

              outputs = model(ids=ids,mask=mask)
              _,preds = torch.max(outputs,dim=1)

              loss = loss_fn(outputs,targets)


              correct_predictions += torch.sum(preds == targets)
              losses.append(loss.item())
             # print(f"Individual Evaluation Losses {loss.item()} " )

      return correct_predictions.double() / n_examples, np.mean(losses)

  def test_fn(model,dataloader,device,):
      model.eval()
      final_pred = []
      with torch.no_grad():
          tk0 = tqdm(test_dataloader,total=len(test_dataloader))
          for data in tk0:
              ids = data['ids'].to(device)
              mask = data['mask'].to(device)
              op = model(ids,mask)
              _,preds = torch.max(op,dim=1)
              # if want the probability dont use max()
              final_pred.append(preds.tolist())
        return final_pred

    
