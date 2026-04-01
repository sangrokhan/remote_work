[FGR-BC0401_tables.md - Table 1]
| Interface / Direction | Sender | Receiver |
| --- | --- | --- |
| S1-U / Downlink | S-GW | CU-UP |
| F1-U / Uplink | DU | CU-UP |
| F1-U / Downlink | CU-UP | DU |

[FGR-BC0401_tables.md - Table 2]
| Interface / Direction | Sender | Receiver |
| --- | --- | --- |
| N3 / Downlink | UPF | CU-UP |
| F1-U / Uplink | DU | CU-UP |
| F1-U / Downlink | CU-UP | DU |

[FGR-BC0401_tables.md - Table 3]
| Parameter | Description |
| --- | --- |
| sequence-number-flag | This parameter indicates whether to configure the GTP sequence number of the GTP Packet.<br>false: The GTP sequence number is not configured to the GTP header information of a GTP packet that is transmitted by a gNB.<br>true: The GTP sequence number is configured to the GTP header information of a GTP packet that is transmitted by a gNB. |

[FGR-BC0401_tables.md - Table 4]
| Parameter | Description |
| --- | --- |
| sgw-gtp-ip-index | This leaf indicates sGW's GTP IP Index. |
| sgw-gtp-ip | This leaf indicates the sGW's GTP IP address. |

[FGR-BC0401_tables.md - Table 5]
| Parameter | Description |
| --- | --- |
| upf-gtp-ip-index | This leaf indicates UPF's GTP IP Index. |
| upf-gtp-ip | This leaf indicates the UPF's GTP IP address. |

[FGR-BC0401_tables.md - Table 6]
| Parameter | Description |
| --- | --- |
| enb-gtp-ip-index | This leaf indicates eNB's GTP IP Index. |
| enb-gtp-ip | This leaf indicates the eNB's GTP IP address. |

[FGR-BC0401_tables.md - Table 7]
| Parameter | Description |
| --- | --- |
| xn-gtp-ip-index | This leaf indicates peer gNB's GTP IP Index. |
| xn-gtp-ip | This leaf indicates the gNB's GTP IP address. |

[FGR-BC0401_tables.md - Table 8]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U UL Interface per QCI | F1UPacketLossCntUL_QC I | This counter is the number of uplink GTP packets that have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1UPacketOosCntUL_QCI | This counter is the number of Out of Sequence downlink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1UPacketCntUL_QCI | This counter is the total number of the uplink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | F1UPacketLossRateUL_Q CI | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |

[FGR-BC0401_tables.md - Table 9]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U UL Interface collected in UP per QCI | F1UPacketLossCntUL_QC I | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1UPacketOosCntUL_QCI | This counter is the number of out-of-sequence downlink packets (that is, the accumulated count of packets whose order of the GTP Sequence Number is reversed) |
|  | F1UPacketCntUL_QCI | This counter is the total number of the uplink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | F1UPacketLossRateUL_Q CI | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |

[FGR-BC0401_tables.md - Table 10]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U UL Interface per UPC | F1UPacketLossCntUL | This counter is the number of uplink GTP packets that have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1UPacketOosCntUL | This counter is the number of out-of-sequence downlink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1UPacketCntUL | This counter is the total number of the uplink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | F1UPacketLossRateUL | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |

[FGR-BC0401_tables.md - Table 11]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U UL Interface collected in UP per UP | F1UPacketLossCntUL | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1UPacketOosCntUL | This counter is the number of Out of Sequence downlink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1UPacketCntUL | This counter is the total number of the uplink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | F1UPacketLossRateUL | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |

[FGR-BC0401_tables.md - Table 12]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U DL Interface per QCI<br>F1-U DL Interface per PRC per QCI | F1UPacketLossCntDL_QC I | This counter is the number of downlink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1UPacketOosCntDL_QCI | This counter is the number of Out of Sequence downlink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1UPacketCntDL_QCI | This counter is the number of the downlink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | F1UPacketLossRateDL_Q CI | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for downlink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |

[FGR-BC0401_tables.md - Table 13]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U DL Interface per DU<br>F1-U DL Interface per PRC per DU | F1UPacketLossCntDL | This counter is the number of downlink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1UPacketOosCntDL | This counter is the number of Out of Sequence downlink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1UPacketCntDL | This counter is the number of the downlink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | F1UPacketLossRateDL | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for downlink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |

[FGR-BC0401_tables.md - Table 14]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| S1-U Interface per sGW IP per QCI | S1URxPacketLossCnt | This counter is the number of the lost downlink GTP packets. A negative number of counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | S1URxPacketOosCnt | This counter is the number of downlink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | S1URxPacketLossRate | This counter is the ratio of lost downlink GTP packets to received downlink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | S1URxPacketCnt | This counter is the total number of received downlink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | S1UTxPacketCnt | This counter is the total number of transmitted uplink GTP packets. |
| F1-U Interface per gNB DU per QCI | F1URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | F1URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1URxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | F1UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |
| X2-U Interface per eNB IP per QCI | X2URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | X2URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | X2URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | X2URxPacketCnt | This counter is the total number of the uplink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | X2UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |

[FGR-BC0401_tables.md - Table 15]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| S1-U Interface collected in UP per sGW IP per QCI | S1URxPacketLossCnt | This counter is the number of the lost downlink GTP packets. A negative number of counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | S1URxPacketOosCnt | This counter is the number of downlink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | S1URxPacketLossRate | This counter is the ratio of lost downlink GTP packets to received downlink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | S1URxPacketCnt | This counter is the total number of received downlink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | S1UTxPacketCnt | This counter is the total number of transmitted uplink GTP packets. |
| F1-U Interface collected in UP per gNB-DU per QCI | F1URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | F1URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, isan out-of-sequence inflow packet for the current collection cycle. |
|  | F1URxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | F1UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |
| X2-U Interface collected in UP per eNB IP per QCI | X2URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | X2URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | X2URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | X2URxPacketCnt | This counter is the total number of the uplink GTP packets received until the time of counting the statistic. Count includes good packets received in sequence and OOS. |
|  | X2UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |

[FGR-BC0401_tables.md - Table 16]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| N3 Interface per UPF IP | N3RxPacketLossCnt | This counter is the number of the lost downlink GTP packets. A negative number of counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | N3RxPacketOosCnt | This counter is the number of downlink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | N3RxPacketLossRate | This counter is the ratio of lost downlink GTP packets to received downlink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | N3RxPacketCnt | This counter is the total number of received downlink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | N3TxPacketCnt | This counter is the total number of transmitted uplink GTP packets. |
| F1-U Interface per gNB DU per 5QI | F1URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | F1URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, isan out-of-sequence inflow packet for the current collection cycle. |
|  | F1URxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | F1UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |

[FGR-BC0401_tables.md - Table 17]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| N3 Interface collected in UP per UPF IP | N3RxPacketLossCnt | This counter is the number of the lost downlink GTP packets. A negative number of counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | N3RxPacketOosCnt | This counter is the number of downlink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | N3RxPacketLossRate | This counter is the ratio of lost downlink GTP packets to received downlink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | N3RxPacketCnt | This counter is the total number of received downlink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | N3TxPacketCnt | This counter is the total number of transmitted uplink GTP packets. |
| F1-U Interface collected in UP per gNB-DU per 5QI | F1URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | F1URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1URxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | F1UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |

[FGR-BC0401_tables.md - Table 18]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| N3 Interface collected in UP per UPF IP | N3RxPacketLossCnt | This counter is the number of the lost downlink GTP packets. A negative number of counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | N3RxPacketOosCnt | This counter is the number of downlink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | N3RxPacketLossRate | This counter is the ratio of lost downlink GTP packets to received downlink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | N3RxPacketCnt | This counter is the total number of received downlink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | N3TxPacketCnt | This counter is the total number of transmitted uplink GTP packets. |
| F1-U Interface collected in UP per gNB-DU per 5QI | F1URxPacketLossCnt | This counter is the number of uplink GTP packets which have been lost until the statistics collection time. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequenceinflow packet for the current collection cycle. |
|  | F1URxPacketOosCnt | This counter is the number of Out of Sequence uplink packets (that is, the accumulated count of packets which order of the GTP Sequence Number is reversed) |
|  | F1URxPacketLossRate | This counter is the ratio of actually lost packets to all packets by the time statistics are collected for uplink GTP packets. A negative value denotes that the packets which were lost in the previous collection cycle, is an out-of-sequence inflow packet for the current collection cycle. |
|  | F1URxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | F1UTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |

[FGR-BC0401_tables.md - Table 19]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| F1-U, XN-U UL Interface collected in UPC per 5QI per SNSSAI<br>F1-U, XN-U UL Interface collected in UPP per 5QI per SNSSAI | PacketLossCntUL | This counter is the number of lost uplink GTP packets. A negative number of the counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | PacketOosCntUL | This counter is the number of uplink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | PacketCntUL | This counter is the total number of received uplink GTP packets. |
|  | PacketLossRateUL | This counter is the ratio of lost uplink GTP packets to received uplink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |

[FGR-BC0401_tables.md - Table 20]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| DL F1-U, Xn-U Interface per PRC per 5QI per S-NSSAI | PacketLossCntDL | This counter is the number of the lost downlink GTP packets. A negative number of the counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | PacketOosCntDL | This counter is the number of downlink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | PacketCntDL | This counter is the total number of received downlink GTP packets. |
|  | PacketLossRateDL | This counter is the ratio of lost downlink GTP packets to received downlink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |

[FGR-BC0401_tables.md - Table 21]
| Family Display Name | Type Name | Type Description |
| --- | --- | --- |
| XN-U Interface per UPC_ID per peer gNB per 5QI | XNURxPacketLossCnt | This counter is the number of lost uplink GTP packets. A negative number of the counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | XNURxPacketOosCnt | This counter is the number of uplink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | XNURxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | XNURxPacketLossRate | This counter is the ratio of lost uplink GTP packets to received uplink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | XNUTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |
| XN-U Interface per UP_ID per peer gNB per 5QI | XNURxPacketLossCnt | This counter is the number of lost uplink GTP packets. A negative number of the counter value indicates that the packets, which were lost in the previous collection cycle, are out-of-sequence inflow packets for the current collection cycle. |
|  | XNURxPacketOosCnt | This counter is the number of uplink GTP packets that are out of sequence in the order of the GTP sequence number. |
|  | XNURxPacketCnt | This counter is the total number of received uplink GTP packets, which include both in-order packets and out-of-sequence packets. |
|  | XNURxPacketLossRate | This counter is the ratio of lost uplink GTP packets to received uplink GTP packets. A negative number of the counter value indicates that the packets lost in the previous collection cycle are out-of-sequence inflow packets for the current collection cycle. |
|  | XNUTxPacketCnt | This counter is the total number of transmitted downlink GTP packets. |
