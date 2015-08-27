__author__ = 'dennis'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


if __name__ == '__main__':
    rawData = []
    ips = []
    sep = [6, 10, 13, 18, 22, 26, 30]
    labels = []

    header = ['f01_total_unique_receivers', 'f02_total_unique_senders', 'f03_inter_receivers_senders',
              'f04_union_receivers_senders', 'f05_ratio_receivers_to_senders', 'f06_avg_freq_sending_pkg',
              'f07_avg_freq_receiving_pkg', 'f08_std_freq_sending_pkg', 'f09_std_freq_receiving_pkg',
              'f10_min_freq_sending_pkg', 'f11_q25_freq_sending_pkg', 'f12_q50_freq_sending_pkg',
              'f13_q75_freq_sending_pkg', 'f14_max_freq_sending_pkg', 'f15_min_freq_receiving_pkg',
              'f16_q25_freq_receiving_pkg', 'f17_q50_freq_receiving_pkg', 'f18_q75_freq_receiving_pkg',
              'f19_max_freq_receiving_pkg', 'f20_ratio_freq_send_receive_pkg', 'f21_corr_freq_send_receive_pkg',
              'f22_total_unique_prot_send', 'f23_total_unique_prot_receiced', 'f24_ratio_unique_prot_send_received',
              'f25_inter_prot_send_received', 'f26_union_prot_send_received', 'f27_total_unique_info_send',
              'f28_total_unique_info_receiced', 'f29_ratio_unique_info_send_received', 'f30_inter_info_send_received',
              'f31_union_info_send_received', 'f00_mac_manufacturer', 'f32_most_freq_send_prot',
              'f33_most_freq_received_prot']

    with open('instances.txt', 'r') as f:
        for line in f:
            ip, ins = line.split(':')
            rawData.append(dict(zip(header, ins.strip().split(', '))))
            ips.append(ip)

    data = pd.DataFrame(rawData, index=ips).convert_objects(convert_numeric=True)

    for ip in ips:
        d = int(ip.split('.')[-1])
        labels.append([s for s, x in enumerate(sep) if x >= d][0])
    pca = PCA(n_components=2)
    pca.fit(data[header[0:31]])
    X = pca.transform(data[header[0:31]])

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()


