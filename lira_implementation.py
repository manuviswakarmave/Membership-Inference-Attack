from collections import defaultdict

import torch.optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pandas as pd
import os
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18



def lira_attack(model, dataset, device, pub_test):

    id_, conf_pred = get_confidence(model, dataset, device)


    id_and_conf =  dict(zip(id_, conf_pred))

    stats_in = get_stats("conf_in.csv")
    stats_out = get_stats("conf_out.csv")

    id_and_scores =normalize_scores(compute_lira_scores(id_and_conf, stats_in, stats_out))   # = normalize_scores()
    for id_, score in id_and_scores.items():
        print(id_, score)

    print("length of the dictionary = ", len(id_and_scores))

    is_valid = validate_submission(id_and_scores, dataset)

    if pub_test:
        compute_tpr_at_fpr(id_and_scores, dataset)

    save_submission(id_and_scores)






def create_conf_csv(dataset, device):


    for i in range (2):
        in_ds, out_ds = create_shadow_split(dataset)

        in_model = train_model(in_ds, device, model_type = "IN")

        ids, confs = get_confidence(in_model, in_ds, device)
        save_conf("conf_in.csv", ids, confs)
        ids, confs = get_confidence(in_model, out_ds, device)
        save_conf("conf_out.csv", ids, confs)


        out_model = train_model(out_ds, device, model_type = "OUT")

        ids, confs = get_confidence(out_model, in_ds, device)
        save_conf("conf_out.csv", ids, confs)
        ids, confs = get_confidence(out_model,out_ds, device)
        save_conf("conf_in.csv", ids, confs)








def create_shadow_split(dataset, seed = None):
    if seed is not None:
        np.random.seed(seed)

    class_indices = defaultdict(list)

    #grouping by labels
    for i in range(len(dataset)):
        _, _, label = dataset[i][:3]
        class_indices[int(label)].append(i)

    in_idx = []
    out_idx = []

    for label, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        split = len(indices) // 2
        in_idx.extend(indices[:split])
        out_idx.extend(indices[split:])

    in_ds = Subset(dataset, in_idx)
    out_ds = Subset(dataset, out_idx)

    return in_ds, out_ds

def train_model(
        ds,
        device,
        model_type,
        epochs = 25,
        batch_size = 32,
        lr = 1e-3,
        weight_decay = 1e-5 ):

    model = create_shadow_model().to(device)

    loader = DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            ids, images, labels = batch[:3]

            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(ds)
        print(f"{model_type } [model : ]Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    return model












def create_shadow_model():
    model = resnet18(weights = None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 9)
    return model


def get_confidence(model, ds, device):
    loader = DataLoader(ds, batch_size = 32, shuffle = False)
    model = model.to(device)
    model.eval()

    ids_all = []
    confs_all = []

    with torch.no_grad():
        for batch in loader:
            ids, images, labels = batch[:3]
            images = images.to(device)
            labels = labels.to(device).long()

            logits = model(images)
            probs = F.softmax(logits, dim = 1)
            conf = probs[
                torch.arange(labels.size(0),
                device = device),
                labels
            ]
            conf = conf.clamp(1e-7, 1 - 1e-7)
            conf = torch.log(conf / (1 - conf))

            ids_all.extend(int(i) for i in ids)
            confs_all.extend(conf.cpu().detach().numpy())

    return ids_all, confs_all

def save_conf(file_name, ids, confs):
    df_new = pd.DataFrame({
        "id" : ids,
        "conf" : confs
    })

    if os.path.exists(file_name):
        df_old = pd.read_csv(file_name)
        df_all = pd.concat([df_old, df_new], ignore_index = True)
    else:
        df_all = df_new

    df_all.to_csv(file_name, index = False)




def get_stats(csv_file):
    df = pd.read_csv(csv_file)

    stats = {}

    for id_, group in df.groupby("id"):
        values = group["conf"].values


        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 1e-6

        stats[id_] = {"mean" : mean, "std" : std}

    return stats



def gaussian_logpdf(x, mean, std):
    std = max(std, 1e-6)  # avoid division by zero
    return -0.5 * np.log(2 * np.pi * std**2) - ((x - mean)**2) / (2 * std**2)


def compute_lira_scores(id_and_conf, stats_in, stats_out):
    id_to_score = {}

    for id_, s in id_and_conf.items():

        # skip if stats missing
        if id_ not in stats_in or id_ not in stats_out:
            id_to_score[id_] = 0.0
            continue

        mu_in = stats_in[id_]["mean"]
        std_in = stats_in[id_]["std"]

        mu_out = stats_out[id_]["mean"]
        std_out = stats_out[id_]["std"]

        logp_in = gaussian_logpdf(s, mu_in, std_in)
        logp_out = gaussian_logpdf(s, mu_out, std_out)

        score = logp_in - logp_out

    # log likelihood ratio

        id_to_score[id_] = score

    return id_to_score




def compute_tpr_at_fpr(id_to_score, dataset, target_fpr=0.05):
    scores = []
    labels = []

    # collect scores + ground truth
    for sample in dataset:
        id_, _, _, membership = sample
        if id_ not in id_to_score:
            print("This id does not exists in the shadow_Conf : ", id_)
            continue

        scores.append(id_to_score[id_])
        labels.append(membership)

    scores = np.array(scores)
    labels = np.array(labels)

    # separate positives and negatives
    pos_scores = scores[labels == 1]  # members
    neg_scores = scores[labels == 0]  # non-members

    # threshold at top 5% of negative scores
    threshold = np.percentile(neg_scores, 100 * (1 - target_fpr))

    # TPR = fraction of positives above threshold
    tpr = np.mean(pos_scores >= threshold)

    print("TPR at 5% FPR : {}".format(tpr))



def normalize_scores(id_to_score):
    sorted_ids = sorted(id_to_score, key=lambda k: id_to_score[k])
    n = len(sorted_ids)
    return {id_: rank / (n - 1) for rank, id_ in enumerate(sorted_ids)}



def validate_submission(id_to_score, dataset):
    errors = []
    required_ids = set(sample[0] for sample in dataset)
    submitted_ids = list(id_to_score.keys())

    # Check for duplicates
    if len(submitted_ids) != len(set(submitted_ids)):
        errors.append("❌ Duplicate IDs found")
    else:
        print("✅ No duplicate IDs")

    # Check no missing IDs
    missing = required_ids - set(submitted_ids)
    extra = set(submitted_ids) - required_ids
    if missing:
        errors.append(f"❌ Missing IDs: {len(missing)} missing")
    else:
        print("✅ No missing IDs")
    if extra:
        errors.append(f"❌ Extra IDs not in dataset: {len(extra)} extra")
    else:
        print("✅ No extra IDs")

    # Check each sample appears exactly once
    if len(submitted_ids) == len(required_ids) and not missing and not extra:
        print("✅ Each sample appears exactly once")

    # Check scores are numeric and in [0, 1]
    invalid_scores = []
    for id_, score in id_to_score.items():
        if not isinstance(score, (int, float, np.floating)):
            invalid_scores.append((id_, score, "not numeric"))
        elif not (0 <= score <= 1):
            invalid_scores.append((id_, score, "out of range"))

    if invalid_scores:
        errors.append(f"❌ Invalid scores: {invalid_scores[:5]}...")  # show first 5
    else:
        print("✅ All scores are numeric and in [0, 1]")

    # Summary
    print("\n--- Validation Summary ---")
    if not errors:
        print("✅ All checks passed! Ready to submit.")
    else:
        for e in errors:
            print(e)

    return len(errors) == 0




def save_submission(id_and_scores, filename="submission.csv"):
    df = pd.DataFrame(list(id_and_scores.items()), columns=["id", "score"])
    df.to_csv(filename, index=False)  # overwrites by default
