import torch
from tqdm import tqdm


def loss_fn(outputs, points):
    return torch.nn.CrossEntropyLoss()( outputs, points )


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm( enumerate( data_loader ), total=len( data_loader ) ):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["points"]

        ids = ids.to( device, dtype=torch.long )
        mask = mask.to( device, dtype=torch.long )
        targets = targets.to( device, dtype=torch.long )

        optimizer.zero_grad()
        outputs = model( ids, attention_mask=mask, labels=targets )
        # loss, output = outputs.loss, outputs.logits
        output = outputs.logits
        loss = loss_fn( output, targets )
        loss.backward()
        torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm( enumerate( data_loader ), total=len( data_loader ) ):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["points"]

            ids = ids.to( device, dtype=torch.long )
            mask = mask.to( device, dtype=torch.long )
            targets = targets.to( device, dtype=torch.long )

            outputs = model( ids, attention_mask=mask, labels=targets )
            output = outputs.logits
            fin_targets.extend( targets.cpu().detach().numpy().tolist())
            fin_outputs.extend( torch.argmax( output, axis=1 ).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
