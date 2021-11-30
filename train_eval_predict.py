import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

## TRAIN
def Train(log_interval, model, device, train_loader, optimizer, scheduler, epoch=1):
    print("================= TRAIN Start =================")
    lossfn = torch.nn.CrossEntropyLoss()
    model.train() 
    
    for batch_idx, datas in enumerate(train_loader):
        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)

        loss = lossfn(output,target)
        # print(loss)
        loss.backward() # loss backprop
        optimizer.step() # optimizer update parameter

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        epoch+=1
    scheduler.step() # scheduler update parameter

## EVALUATE
def valid_EVAL(model, device, test_loader):
    print("================= EVAL Start =================")
    model.eval() 
    test_loss = []
    correct = []
    lossfn = torch.nn.CrossEntropyLoss()

    preds=[]
    targets=[]
    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
            #print(target)
            outputs=model(data)
            test_loss.append(lossfn(outputs, target).item()) # sum up batch loss
            pred = outputs.argmax(dim=1,keepdim=True) # get the index of the max probability 인덱스            
            # print(outputs, pred,target.data)
            correct.append(pred.eq(target.data.view_as(pred)).sum().item())  

            preds.extend(outputs.argmax(dim=1,keepdim=False).cpu().numpy())
            targets.extend(target.cpu().numpy())

    loss = sum(test_loss)/len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Balanced Accuracy: {:.4f}% '
        .format(loss, sum(correct), len(test_loader.dataset), 100. * sum(correct) / len(test_loader.dataset), 100. * balanced_accuracy_score(targets,preds)))
    return loss, 100. * sum(correct) / len(test_loader.dataset), 100. * balanced_accuracy_score(targets,preds)

## test_prediction
def predict_EVAL(model, device, test_loader):
    print("================= EVAL Start =================")
    model.eval() 

    preds=[]
    with torch.no_grad():
        for datas in test_loader:
            data = datas[0].to(device)
            #print(target)
            outputs=model(data)
            preds.extend(outputs.argmax(dim=1,keepdim=False).cpu().numpy())# get the index of the max probability 인덱스            
    
    return preds