import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from  torch.utils.data import DataLoader



BASE = Path(__file__).parent

def run_naive_attack(model, priv_data, device, batch_size = 64):
    model.to(device)
    model.eval()

    sample = priv_data[0]
    print(sample)

    loader = DataLoader(priv_data, batch_size = batch_size, shuffle = False)

    all_ids = []
    all_scores = []

    with torch.no_grad():
        for id_, img, label in loader:
            img = img.to(device)
            logits = model(img)
            probs = F.softmax(logits, dim = 1)
            scores = probs.max(dim = 1).values

            all_ids.extend(id_.tolist())
            all_scores.extend(scores.tolist())


    df = pd.DataFrame({'id': all_ids, 'score': all_scores})
    df.to_csv(BASE / 'scores.csv', index = False)
