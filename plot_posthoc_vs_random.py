import argparse
import random
from pathlib import Path

import numpy as np
import torch

from dataset import load_data
from models import EdgeNetAndRejector, LocalNet
from test_utils import get_accuracy, get_accuracy_remote, test_edge, test_local, get_accuracy_remote_random,get_accuracy_remote_bounded_reject_rate_stochastic

if __name__ == '__main__':
    output_stochastic=[]
    output_last_stochastic=0
    output_random=[]
    output_last_random=0
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="server model to use")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to use")
    parser.add_argument("--save_dir", type=str, default="models", help="directory to saved client model")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset


    # dataset pre-prossessing
    Batch_size=16
    Batch_size_test=16
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    #device = torch.device("cpu")
    print(f"Using {device} device")

    trainset_full, testset_full = load_data(dataset)

    # Check if the testset is not empty
    if len(testset_full) == 0:
        raise ValueError("The test dataset is empty!")

    # Create data loaders
    testloader = torch.utils.data.DataLoader(
        testset_full,
        batch_size=Batch_size_test,
        shuffle=False,
        num_workers=2
    )


    mobile_net = LocalNet(len(trainset_full.classes)).to(device)
    #
    if device.type !="cuda":
        mobile_net.load_state_dict( torch.load(save_dir / f'{dataset}-localnet.pth', map_location=torch.device('cpu') ))
    else:
        mobile_net.load_state_dict(torch.load(save_dir / f'{dataset}-localnet.pth'))
    mobile_net.eval()
    #

    print("------------------ Only run on local ------------------")
    labels, predicted_local = test_local(mobile_net, testloader)
    total, correct, labels_map = get_accuracy(predicted_local, labels)
    print("Sample size: ", total)
    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
    output_stochastic.append(1.0*correct / total)
    output_random.append(1.0*correct / total)
    # print('Accuracy for each class:')
    # for i, class_name in enumerate(testset_full.classes):
    #     print(f'{class_name}: {labels_map[i]["correct"] / labels_map[i]["total"] * 100:.2f}%')
    #

    edge_net = EdgeNetAndRejector(len(trainset_full.classes)).to(device)
    if device.type != "cuda":
        edge_net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    else:
        edge_net.load_state_dict(torch.load(args.model))
    edge_net.eval();

    print("------------------ Only run on edge ------------------")
    labels, predicted_edge_r, predicted_edge_e = test_edge(edge_net, testloader)
    # only run on edge
    total, correct, labels_map = get_accuracy(predicted_edge_e, labels)
    print("Sample size: ", total)
    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
    output_last_stochastic=1.0*correct / total
    output_last_random=1.0*correct / total
    # print('Accuracy for each class:')
    # for i, class_name in enumerate(testset_full.classes):
    #     print(f'{class_name}: {labels_map[i]["correct"] / labels_map[i]["total"] * 100:.2f}%')


    print("------------------ Run on both edge and local ------------------")
    total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote(predicted_local, predicted_edge_r, predicted_edge_e, labels)
    print('Sample size: %d' %total)
    print(f"Decision made locally: {predict_locally}")
    print(f"Decision made on edge: {predict_remotely}")
    estimated_reject_rate= predict_remotely/total
    print('Accuracy of the network on test images: %d%%' % (
        100 * correct / total))
    # print('Accuracy for each class:')
    # for i, class_name in enumerate(testset_full.classes):
    #     local_acc = labels_map[i]["local_correct"] / labels_map[i]["local_total"] if labels_map[i]["local_total"] > 0 else float("nan")
    #     remote_acc = labels_map[i]["remote_correct"] / labels_map[i]["remote_total"] if labels_map[i]["remote_total"] > 0 else float("nan")
    #     print(
    #         f'{class_name}:\n\t'
    #         f'Local Total: {labels_map[i]["local_total"]}, Edge Total: {labels_map[i]["remote_total"]}\n'
    #         f"\tLocal: {local_acc * 100:.2f}%, "
    #         f"Edge: {remote_acc * 100:.2f}%, "
    #         f"Overall: {(labels_map[i]['local_correct'] + labels_map[i]['remote_correct']) / (labels_map[i]['local_total'] + labels_map[i]['remote_total']) * 100:.2f}%"
    #     )


    print("------------------ Run on both edge and local with stochastic bounded reject rate------------------")
    for k in range(1, 10):
        reject_rate = k*1.0 / 10

        total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote_bounded_reject_rate_stochastic(predicted_local, predicted_edge_r, predicted_edge_e, labels, reject_rate=reject_rate,estimated_reject_rate=estimated_reject_rate)
        print('Sample size: %d' %total)
        print(f"reject rate: {reject_rate}")
        print(f"Decision made locally: {predict_locally}")
        print(f"Decision made on edge: {predict_remotely}")

        print('Accuracy of the network on test images: %d%%' % (
            100 * correct / total))
        output_stochastic.append(1.0*correct / total)
        # print('Accuracy for each class:')

    print("------------------ Run on both edge and local with random rate (without referring to rejector)------------------")
    for k in range(1, 10):
        reject_rate = k*1.0 / 10
        total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote_random(predicted_local, predicted_edge_r, predicted_edge_e, labels, reject_rate=reject_rate)
        # print('Sample size: %d' %total)
        print(f"reject rate: {reject_rate}")
        print(f"Decision made locally: {predict_locally}")
        print(f"Decision made on edge: {predict_remotely}")

        print('Accuracy of the network on test images: %d%%' % (
            100 * correct / total))
        output_random.append(1.0*correct / total)
        # print('Accuracy for each class:')
        # for i, class_name in enumerate(testset_full.classes):
        #     local_acc = labels_map[i]["local_correct"] / labels_map[i]["local_total"] if labels_map[i]["local_total"] > 0 else float("nan")
        #     remote_acc = labels_map[i]["remote_correct"] / labels_map[i]["remote_total"] if labels_map[i]["remote_total"] > 0 else float("nan")
        #     print(
        #         f'{class_name}:\n\t'
        #         f'Local Total: {labels_map[i]["local_total"]}, Edge Total: {labels_map[i]["remote_total"]}\n'
        #         f"\tLocal: {local_acc * 100:.2f}%, "
        #         f"Edge: {remote_acc * 100:.2f}%, "
        #         f"Overall: {(labels_map[i]['local_correct'] + labels_map[i]['remote_correct']) / (labels_map[i]['local_total'] + labels_map[i]['remote_total']) * 100:.2f}%"
        #     )
    output_stochastic.append(output_last_stochastic)
    output_random.append(output_last_random)
    print(output_stochastic)
    print(output_random)