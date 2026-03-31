[FGR-BC0201_tables.md - Table 1]
| Parameter | Description |
| --- | --- |
| srb-id | The ID of SRB to retrieve. 1: Information on SRB1. 2: Information on SRB2.<br>3: Information on SRB3 for NSA. |
| gnb-timer-poll-retransmit | This parameter is the gNB timer to retransmit the poll in a transmitting AM RLC entity. |
| ue-timer-poll-retransmit | This parameter is the UE timer to retransmit the poll in a transmitting AM RLC entity. |
| gnb-poll-pdu | This parameter is the threshold to trigger the poll for pollPDU in an AM RLC entity at gNB side. |
| ue-poll-pdu | This parameter is the threshold to trigger the poll for pollPDU in an AM RLC entity at UE side. |
| gnb-poll-byte | This parameter is the threshold used to trigger the poll for pollByte in an AM_RLC entity at gNB side. |
| ue-poll-byte | This parameter is the threshold used to trigger the poll for pollByte in an AM_RLC entity at UE side. |
| gnb-max-retransmission-threshold | This parameter is the threshold used to limit the number of the AMD PDU retransmission in a transmitting AM_RLC entity at gNB side. |
| ue-max-retransmission-threshold | This parameter is the threshold used to limit the number of the AMD PDU retransmission in a transmitting AM_RLC entity at UE side. |
| gnb-t-reassembly | This parameter is the timer to reassemble of RLC PDUs in a receiving RLC entity at gNB side. |
| ue-t-reassembly | This parameter is the timer to reassemble of RLC PDUs in a receiving RLC entity at UE side. |
| gnb-timer-status-prohibit | This parameter is the timer to prohibit the transmission of STATUS_PDU in a receiving AM_RLC entity at gNB side. |
| ue-timer-status-prohibit | This parameter is the timer to prohibit the transmission of STATUS_PDU in a receiving AM_RLC entity at UE side. |
| sn-field-length-ul-am | RLC AM mode SN field length. |

[FGR-BC0201_tables.md - Table 2]
| Parameter | Description |
| --- | --- |
| qci | This parameter is the QoS Class Identifier(QCI).<br>The standard QCI defined in the standard is 1 to 9. 0 and 10 to 255 can be used by an operator. |
| gnb-timer-poll-retransmit | This parameter is the timer to retransmit the poll in a transmitting AM RLC entity. |
| gnb-poll-pdu | This parameter is the threshold to trigger the poll for pollPDU in an AM RLC entity. |
| gnb-poll-byte | This parameter is the threshold used to trigger the poll for pollByte in an AM_RLC entity. |
| gnb-max-retransmission-threshold | This parameter is the threshold used to limit the number of AMD PDU retransmissions in a transmitting AM_RLC entity. |
| gnb-t-reassembly | This parameter is the timer to reassemble of RLC PDUs in a receiving RLC entity |
| gnb-timer-status-prohibit | This parameter is the timer to prohibit the transmission of STATUS_PDU in a receiving AM_RLC entity. |
| ue-timer-poll-retransmit | This parameter is the timer to retransmit the poll in a UE transmitting the AM RLC entity. |
| ue-poll-pdu | This parameter is the threshold to trigger the poll for pollPDU in an UE AM RLC entity. |
| ue-poll-byte | This parameter is the threshold used to trigger the poll for pollByte in an UE side AM_RLC entity. |
| ue-max-retransmission-threshold | This parameter is the threshold used to limit the number of the AMD PDU retransmission in a UE side transmitting AM_RLC entity. |
| ue-t-reassembly | This parameter is the timer to reassemble of RLC PDUs in a UE side receiving the RLC entity. |
| ue-timer-status-prohibit | This parameter is the timer to prohibit the transmission of STATUS_PDU in the UE side receiving the AM_RLC entity. |
| sn-field-length-ul-um | This parameter is the RLC configuration of UM mode per QCI for the sequence number size of uplink UM RLC entity(6 bit or 12 bit) |
| sn-field-length-ul-am | This parameter is the RLC Configurations of AM mode per QCI for the sequence number size of uplink AM RLC entity(12 bit or 18 bit) |
| sn-field-length-dl-um | This parameter is the RLC Configurations of UM mode per QCI for the sequence number size of downlink UM RLC entity(6 bit or 12 bit) |
| sn-field-length-dl-am | This parameter is the RLC Configurations of AM mode per QCI for the sequence number size of downlink AM RLC entity(12 bit or 18 bit) |

[FGR-BC0201_tables.md - Table 3]
| Parameter | Description |
| --- | --- |
| qos-5qi | This parameter is the 5G QoS Identifier(5QI).<br>The standard 5QI defined in standard is 1 to 9. 0 and 10 to 255 can be used by an operator. |
| gnb-timer-poll-retransmit | This parameter is the timer to retransmit the poll in a transmitting AM RLC entity. |
| gnb-poll-pdu | This parameter is the threshold to trigger the poll for pollPDU in an AM RLC entity. |
| gnb-poll-byte | This parameter is the threshold used to trigger the poll for pollByte in an AM_RLC entity. |
| gnb-max-retransmission-threshold | This parameter is the threshold used to limit the number of the AMD PDU retransmission in a transmitting AM_RLC entity. |
| gnb-t-reassembly | This parameter is the timer to reassemble of RLC PDUs in a receiving RLC entity |
| gnb-timer-status-prohibit | This parameter is the timer to prohibit the transmission of STATUS_PDU in a receiving AM_RLC entity. |
| ue-timer-poll-retransmit | This parameter is the timer to retransmit poll in a UE transmitting the AM RLC entity. |
| ue-poll-pdu | This parameter is the threshold to trigger the poll for pollPDU in an UE AM RLC entity. |
| ue-poll-byte | This parameter is the threshold used to trigger the poll for pollByte in a UE side AM_RLC entity. |
| ue-max-retransmission-threshold | This parameter is the threshold used to limit the number of AMD PDU retransmission in a UE side transmitting AM_RLC entity. |
| ue-t-reassembly | This parameter is the timer to reassemble of RLC PDUs in a UE side receiving the RLC entity. |
| ue-timer-status-prohibit | This parameter is the timer to prohibit the transmission of STATUS_PDU in a UE side receiving the AM_RLC entity. |
| sn-field-length-ul-um | This parameter is the RLC configuration of UM mode per QCI for sequence number size of uplink UM RLC entity(6 bit or 12 bit) |
| sn-field-length-ul-am | This parameter is the RLC Configurations of AM mode per QCI for the sequence number size of uplink AM RLC entity(12 bit or 18 bit) |
| sn-field-length-dl-um | This parameter is the RLC Configurations of UM mode per QCI for sequence number size of downlink UM RLC entity(6 bit or 12 bit) |
| sn-field-length-dl-am | This parameter is the RLC Configurations of UM mode per QCI for sequence number size of downlink AM RLC entity(12 bit or 18 bit) |

[FGR-BC0201_tables.md - Table 4]
| Parameter | Description |
| --- | --- |
| adaptive-rlc-poll-retx-dl-enable | This parameter is used to enable/disable the adaptive RLC poll retransmission function. |
| adaptive-retx-point-dl | This parameter is used for the configuration of the adaptive retransmission timer application point. The adaptive timer value starts to be applied from the number of retransmissions of this setting value. |
| adaptive-first-rlc-poll-timer | This parameter is used to configure the poll timer that is first applied during adaptive retransmission. |
| adaptive-second-rlc-poll-timer | This parameter is used to configure the poll timer that is applied second during adaptive retransmission. |
| adaptive-last-rlc-poll-timer | This parameter is used to configure the poll timer that is lastly applied at the adaptive retransmission point. |
