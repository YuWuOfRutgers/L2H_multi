import torch
import numpy as np
import os
from torchvision import transforms

def bernoulli_random(p):
    return 1 if np.random.uniform(0,1) < p else 0

def test_local(mobile_net, testloader):
    predicted = []
    labels_all = []
    device = next(mobile_net.parameters()).device

    mobile_net.eval()

    with torch.no_grad():
        #print(len(testloader))
        for images, labels in testloader:
            #print("1")
            images, labels = images.to(device), labels.to(device)
            #print("2")
            local_outputs = mobile_net(images)

            predicted.append(local_outputs)
            labels_all.append(labels)

    predicted = torch.cat(predicted)
    labels_all = torch.cat(labels_all)
    return labels_all, predicted


def test_edge(edge_net, testloader):
    predicted_edge_r = []
    predicted_edge_e = []
    labels_all = []
    device = next(edge_net.parameters()).device
    edge_net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            edge_outputs_r, edge_outputs_e = edge_net(images)
            predicted_edge_r.append(edge_outputs_r)
            predicted_edge_e.append(edge_outputs_e)
            labels_all.append(labels)

    predicted_edge_r = torch.cat(predicted_edge_r)
    predicted_edge_e = torch.cat(predicted_edge_e)
    labels_all = torch.cat(labels_all)
    return labels_all, predicted_edge_r, predicted_edge_e

def test_edge_save_images_according_to_rejector(edge_net, testloader):
    predicted_edge_r = []
    predicted_edge_e = []
    labels_all = []
    device = next(edge_net.parameters()).device
    edge_net.eval()
    local_folder = 'local_images'
    edge_folder = 'edge_images'
    os.makedirs(local_folder, exist_ok=True)
    os.makedirs(edge_folder, exist_ok=True)
    local_image_index= 0
    remote_image_index= 0
    with torch.no_grad():
        for raw_images, labels in testloader:
            images, labels = raw_images.to(device), labels.to(device)

            edge_outputs_r, edge_outputs_e = edge_net(images)
            # print(edge_outputs_r)
            # print(edge_outputs_r[0])
            # print(edge_outputs_r[0][0])
            # print(edge_outputs_r[0][1])
            if edge_outputs_r[0][0]< edge_outputs_r[0][1]: #if r(x) is remote
                image_pil = transforms.ToPILImage()(raw_images[0])
                file_name = f"{edge_folder}/{labels[0]}_image_{remote_image_index}.png"
                image_pil.save(file_name)
                remote_image_index += 1

            elif edge_outputs_r[0][0] >= edge_outputs_r[0][1]: # if r(x) is local
                image_pil = transforms.ToPILImage()(raw_images[0])
                file_name = f"{local_folder}/{labels[0]}_image_{local_image_index}.png"
                image_pil.save(file_name)        
                local_image_index += 1      

            predicted_edge_r.append(edge_outputs_r)
            predicted_edge_e.append(edge_outputs_e)
            labels_all.append(labels)

    predicted_edge_r = torch.cat(predicted_edge_r)
    predicted_edge_e = torch.cat(predicted_edge_e)
    labels_all = torch.cat(labels_all)
    return labels_all, predicted_edge_r, predicted_edge_e


def get_accuracy(predicted, labels):
    predicted_label = torch.argmax(predicted, dim=1)
    classes = labels.unique()

    labels_map = {i: {"correct": 0, "total": 0} for i in range(len(classes))}

    for cls in classes:
        labels_map[cls.item()]["total"] = (labels == cls).sum().item()
        labels_map[cls.item()]["correct"] = ((labels == cls) & (predicted_label == cls)).sum().item()

    total = sum([v["total"] for v in labels_map.values()])
    correct = sum([v["correct"] for v in labels_map.values()])

    return total, correct, labels_map


def get_accuracy_remote(predicted_local, predicted_edge_r, predicted_edge_e, labels):
    is_remote = torch.argmax(predicted_edge_r, dim=1)
    #print((is_remote).shape)
    classes = labels.unique()

    labels_map = {i: {
        "local_correct": 0, "local_total": 0,
        "remote_correct": 0, "remote_total": 0
    } for i in range(len(classes))}
    predicted = predicted_local * (1-is_remote.unsqueeze(1)) + predicted_edge_e * is_remote.unsqueeze(1)
    predicted_label = torch.argmax(predicted, dim=1)

    for cls in classes:
        labels_map[cls.item()]["local_total"] = ((labels == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_total"] = ((labels == cls) & is_remote).sum().item()
        labels_map[cls.item()]["local_correct"] = ((labels == cls) & (predicted_label == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_correct"] = ((labels == cls) & (predicted_label == cls) & is_remote).sum().item()

    total = sum([v["local_total"] + v["remote_total"] for v in labels_map.values()])
    correct = sum([v["local_correct"] + v["remote_correct"] for v in labels_map.values()])
    predict_locally = sum([v["local_total"] for v in labels_map.values()])
    predict_remotely = sum([v["remote_total"] for v in labels_map.values()])

    return total, correct, predict_locally, predict_remotely, labels_map


def get_accuracy_remote_bounded_reject_rate(predicted_local, predicted_edge_r, predicted_edge_e, labels, reject_rate=0.5):
    is_remote = torch.argmax(predicted_edge_r, dim=1)
    print((is_remote).shape)
    #add sliding window to here:
    gap = int(1/reject_rate)
    i = 0
    while i < len(is_remote):
        if is_remote[i] == 1:# 1 means remote
            for j in range(1, gap):
                if i+j<len(is_remote):
                    is_remote[i+j] = 0
                else:
                    break
            i += gap
        else:
            i += 1

    classes = labels.unique()

    labels_map = {i: {
        "local_correct": 0, "local_total": 0,
        "remote_correct": 0, "remote_total": 0
    } for i in range(len(classes))}
    predicted = predicted_local * (1-is_remote.unsqueeze(1)) + predicted_edge_e * is_remote.unsqueeze(1)
    predicted_label = torch.argmax(predicted, dim=1)

    for cls in classes:
        labels_map[cls.item()]["local_total"] = ((labels == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_total"] = ((labels == cls) & is_remote).sum().item()
        labels_map[cls.item()]["local_correct"] = ((labels == cls) & (predicted_label == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_correct"] = ((labels == cls) & (predicted_label == cls) & is_remote).sum().item()

    total = sum([v["local_total"] + v["remote_total"] for v in labels_map.values()])
    correct = sum([v["local_correct"] + v["remote_correct"] for v in labels_map.values()])
    predict_locally = sum([v["local_total"] for v in labels_map.values()])
    predict_remotely = sum([v["remote_total"] for v in labels_map.values()])

    return total, correct, predict_locally, predict_remotely, labels_map


def get_accuracy_remote_bounded_reject_rate_stochastic(predicted_local, predicted_edge_r, predicted_edge_e, labels, reject_rate=0.5,estimated_reject_rate=0.4):
    is_remote = torch.argmax(predicted_edge_r, dim=1)
    #print((is_remote).shape)
    #add stochastic adjuster to here:
    if estimated_reject_rate > reject_rate:

        ratio= reject_rate/estimated_reject_rate
        print(ratio)
        i = 0
        while i < len(is_remote):
            if is_remote[i] == 1:# 1 means remote
                is_remote[i] = bernoulli_random(ratio)
            i += 1
    else:
        extra_reject=reject_rate-estimated_reject_rate
        ratio=extra_reject/(1-estimated_reject_rate)
        i = 0
        while i< len(is_remote):
            if is_remote[i] == 0:
                is_remote[i] = bernoulli_random(ratio)
            i += 1
    classes = labels.unique()

    labels_map = {i: {
        "local_correct": 0, "local_total": 0,
        "remote_correct": 0, "remote_total": 0
    } for i in range(len(classes))}
    predicted = predicted_local * (1-is_remote.unsqueeze(1)) + predicted_edge_e * is_remote.unsqueeze(1)
    predicted_label = torch.argmax(predicted, dim=1)

    for cls in classes:
        labels_map[cls.item()]["local_total"] = ((labels == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_total"] = ((labels == cls) & is_remote).sum().item()
        labels_map[cls.item()]["local_correct"] = ((labels == cls) & (predicted_label == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_correct"] = ((labels == cls) & (predicted_label == cls) & is_remote).sum().item()

    total = sum([v["local_total"] + v["remote_total"] for v in labels_map.values()])
    correct = sum([v["local_correct"] + v["remote_correct"] for v in labels_map.values()])
    predict_locally = sum([v["local_total"] for v in labels_map.values()])
    predict_remotely = sum([v["remote_total"] for v in labels_map.values()])

    return total, correct, predict_locally, predict_remotely, labels_map


#randomly reject on any sample with the parameter reject_rate
def get_accuracy_remote_random(predicted_local, predicted_edge_r, predicted_edge_e, labels, reject_rate=0.5):
    is_remote = torch.argmax(predicted_edge_r, dim=1)
    print((is_remote).shape)
    #add sliding window to here:
    # gap = int(1/reject_rate)
    # i = 0
    # while i < len(is_remote):
    #     is_remote[i] = 1
    #     for j in range(1, gap):
    #         if i+j<len(is_remote):
    #             is_remote[i+j] = 0
    #         else:
    #             break
    #     i += gap
    i = 0
    while i < len(is_remote):
        is_remote[i] = bernoulli_random(reject_rate)
        i += 1

    classes = labels.unique()

    labels_map = {i: {
        "local_correct": 0, "local_total": 0,
        "remote_correct": 0, "remote_total": 0
    } for i in range(len(classes))}
    predicted = predicted_local * (1-is_remote.unsqueeze(1)) + predicted_edge_e * is_remote.unsqueeze(1)
    predicted_label = torch.argmax(predicted, dim=1)

    for cls in classes:
        labels_map[cls.item()]["local_total"] = ((labels == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_total"] = ((labels == cls) & is_remote).sum().item()
        labels_map[cls.item()]["local_correct"] = ((labels == cls) & (predicted_label == cls) & (1-is_remote)).sum().item()
        labels_map[cls.item()]["remote_correct"] = ((labels == cls) & (predicted_label == cls) & is_remote).sum().item()

    total = sum([v["local_total"] + v["remote_total"] for v in labels_map.values()])
    correct = sum([v["local_correct"] + v["remote_correct"] for v in labels_map.values()])
    predict_locally = sum([v["local_total"] for v in labels_map.values()])
    predict_remotely = sum([v["remote_total"] for v in labels_map.values()])

    return total, correct, predict_locally, predict_remotely, labels_map

