import os
import torch
from sklearn.metrics import r2_score

def _calc_loss_r2(x, y, model, loss_fn, r2):
    label = torch.flatten(y.T)
    # Compute prediction error
    logits, prediction = model(x.float())
    loss = loss_fn(logits, label)
    y_pred = torch.argmax(prediction, dim=-1).T
    r2 += r2_score(
        y.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy()
    )
    return loss, r2

def _save_model(model, region, type_model, path, best=False):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, f'{type_model}_region_{region}.pt')
    if best == True:
        filename = os.path.join(path, f'Best_{type_model}_region_{region}.pt')
    torch.save(model.state_dict(), filename)

def _get_device(gpu=False):
    if gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cpu': 
            print("You don't have GPU to impute genotype, so I will run by CPU")
        else:
            print(f"You're using GPU {torch.cuda.get_device_name(0)} to impute genotype")
    else: 
        device = 'cpu'
        print("You're using CPU to impute genotype")
    return device