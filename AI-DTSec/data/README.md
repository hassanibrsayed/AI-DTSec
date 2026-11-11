# Data

The datasets are not redistributed in this repository. Download them from their owners and place them as follows.

## WUSTL-EHMS-2020

Washington University in St. Louis: https://research.engineering.wustl.edu/~jain/ehms/index.html

Save the CSV file (3,946,888 bytes) as

    data/wustl-ehms-2020.csv

or set the environment variable `WUSTL_CSV` to its path.

## CICIoMT2024

Canadian Institute for Cybersecurity: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html (registration required)

Only the WiFi-and-MQTT CSV files of the eight classes below are used. Keep the `train` and `test` folders:

    data/CICIoMT2024/train/<Class>_train.pcap.csv
    data/CICIoMT2024/test/<Class>_test.pcap.csv

Classes: `Benign`, `ARP_Spoofing`, `MQTT-Malformed_Data`, `MQTT-DoS-Connect_Flood`, `Recon-OS_Scan`, `Recon-Ping_Sweep`, `Recon-Port_Scan`, `Recon-VulScan`.

Or set the environment variable `CIC_DIR` to the folder that contains `train` and `test`.
