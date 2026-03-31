## FGR-BC0401, Packet loss detection over GTP interface
### INTRODUCTION
This feature provides statistics about lost packets and out-of-sequence packets in GTP interfaces. This feature enables a gNB detect lost packets and out-of-sequence packets over GTP interfaces. For this purpose, the sending side transmits GTP packets after marking a sequence number to help count lost packets at the receiving side. The receiving side measures the quality of the GTP interface through the received GTP packet.
### BENEFIT
 The quality of the backhaul network can be known.
 The quality of the mid-haul network can be known.
### RELEASE HISTORY
 SVR24A
Description
o In the NSA system, GTP Loss/OOS counters for DL F1-U interface per PRC_ID/QCI is supported.
o In the SA system, GTP Loss/OOS counters for DL F1-U interface per PRC_ID is supported.
o In the SA system, GTP Loss/OOS counters for UL F1-U/Xn-U interface per UPC_ID/5QI/S-NSSAI is supported. (for VNF)
o In the SA system, GTP Loss/OOS counters for DL F1-U/Xn-U interface per PRC_ID/5QI/S-NSSAI is supported.
Enhancement
o Before Change
 In the NSA system, GTP Loss/OOS counters for DL F1-U interface per PRC_ID/QCI is not supported.
 In the SA system, GTP Loss/OOS counters for DL F1-U interface per PRC_ID are not supported.
 In the SA system, GTP Loss/OOS counters for UL F1-U/Xn-U interface per UPC_ID/5QI/S-NSSAI is not supported. (for VNF)
 In the SA system, GTP Loss/OOS counters for DL F1-U/Xn-U interface per PRC_ID/5QI/S-NSSAI is not supported.

o After Change
 In the NSA system, GTP Loss/OOS counters for DL F1-U interface per PRC_ID/QCI is supported.
 In the SA system, GTP Loss/OOS counters for DL F1-U interface per PRC_ID is supported.
 In the SA system, GTP Loss/OOS counters for UL F1-U/Xn-U interface per UPC_ID/5QI/S-NSSAI is supported. (for VNF)
 In the SA system, GTP Loss/OOS counters for DL F1-U/Xn-U interface per PRC_ID/5QI/S-NSSAI is supported.
 A new family is added.
 F1-U DL Interface per PRC per QCI
 F1-U DL Interface per PRC per DU
 F1-U, XN-U UL Interface collected in UPC per 5QI per SNSSAI
 DL F1-U, Xn-U Interface per PRC per 5QI per S-NSSAI
 SVR24B
Description
o In the NSA system, GTP Loss/OOS counters for UL F1-U interface per UP_ID/QCI is supported. (for CNF)
o In the SA system, GTP Loss/OOS counters for UL F1-U/Xn-U interface per UP_ID/5QI/S-NSSAI is supported. (for CNF)
Enhancement
o Before Change
 In the NSA system, GTP Loss/OOS counters for UL F1-U interface per UP_ID/QCI is not supported. (for CNF)
 In the SA system, GTP Loss/OOS counters for UL F1-U/Xn-U interface per UP_ID/5QI/S-NSSAI is not supported. (for CNF)
o After Change
 In the NSA system, GTP Loss/OOS counters for UL F1-U interface per UP_ID/QCI is supported. (for CNF)
 In the SA system, GTP Loss/OOS counters for UL F1-U/Xn-U interface per UP_ID/5QI/S-NSSAI is supported. (for CNF)
A new family is added.
 F1-U UL Interface collected in UP per QCI
 F1-U, XN-U UL Interface collected in UPP per 5QI per SNSSAI
 SVR25B
Description

(For VM_CU)
o In the NSA system, GTP Loss/OOS counters for F1-U Interface per gNB DU per QCI is supported.
o In the NSA system, GTP Loss/OOS counters for X2-U Interface per eNB IP per QCI is supported.
o In the NSA system, GTP Loss/OOS counters for S1-U Interface per sGW IP per QCI is supported.
o In the SA system, GTP Loss/OOS counters for F1-U Interface per gNB DU per 5QI is supported.
o In the SA system, GTP Loss/OOS counters for N3 Interface per UPF IP is supported.
o In the SA system, GTP Loss/OOS counters for XN-U Interface per UPC_ID per peer gNB per 5QI is supported.
(For C_CU)
o In the NSA system, GTP Loss/OOS counters for F1-U Interface collected in UP per gNB-DU per QCI is supported.
o In the NSA system, GTP Loss/OOS counters for X2-U Interface collected in UP per eNB IP per QCI per QCI is supported.
o In the NSA system, GTP Loss/OOS counters for S1-U Interface collected in UP per sGW IP per QCI is supported.
o In the SA system, GTP Loss/OOS counters for F1-U Interface collected in UP per gNB-DU per 5QI is supported.
o In the SA system, GTP Loss/OOS counters for N3 Interface collected in UP per UPF IP is supported.
o In the SA system, GTP Loss/OOS counters for XN-U Interface per UP_ID per peer gNB per 5QI is supported.
Enhancement
o Before Change: There is no additional counters.
o After Change
(For VM_CU)
 In the NSA system, GTP Loss/OOS counters for F1-U Interface per gNB DU per QCI is supported.
 In NSA system, GTP Loss/OOS counters for X2-U Interface per eNB IP per QCI is supported.
 In the NSA system, GTP Loss/OOS counters for S1-U Interface per sGW IP per QCI is supported.
 In the SA system, GTP Loss/OOS counters for F1-U Interface per gNB DU per 5QI is supported.
 In the SA system, GTP Loss/OOS counters for N3 Interface per UPF IP is supported.
 In the SA system, GTP Loss/OOS counters for XN-U Interface per

UPC_ID per peer gNB per 5QI is supported.
(For C_CU)
 In the NSA system, GTP Loss/OOS counters for F1-U Interface collected in UP per gNB-DU per QCI is supported.
 In the NSA system, GTP Loss/OOS counters for X2-U Interface collected in UP per eNB IP per QCI per QCI is supported.
 In the NSA system, GTP Loss/OOS counters for S1-U Interface collected in UP per sGW IP per QCI is supported.
 In the SA system, GTP Loss/OOS counters for F1-U Interface collected in UP per gNB-DU per 5QI is supported.
 In the SA system, GTP Loss/OOS counters for N3 Interface collected in UP per UPF IP is supported.
 In the SA system, GTP Loss/OOS counters for XN-U Interface per UP_ID per peer gNB per 5QI is supported.
### DEPENDENCY
 NE Dependency: DU/RU RAN(UP) SGW(4G) UPF(5G)
 Development Dependency: RAN (UP)
 Others
o [NSA system] S-GW should support GTP sequence numbering for S1-U downlink.
o [SA system] UPF should support GTP sequence numbering for N3 downlink.
### LIMITATION
None
### SYSTEM IMPACT
The implementation of this feature has no impact on the network.

### FEATURE DESCRIPTION
#### Overview & Operation
Non Stand Alone
There are two external networks of NSA virtualization systems. One of them is the backhaul network between Core and gNB-UP, which is connected by the S1 interface. The other is mid-haul network between gNB-UP and DU, and is connected by F1 interface. To measure network quality, a gNB collects the statistics for the number of lost packets and the number of out-of-sequence packets for user packets in the GTP layer. For this purpose, the sending side adds a sequence number to each GTP packet for transmitting. The receiving side checks the sequence number for a received packet. If a packet with the specific sequence number is not received, it is judged as a lost packet. If there is a packet with a changed order due to the sequence number, the out-of-sequence packet count is increased. The following figure depicts what statistics items CU-UP and DU can collect.

The following table outlines Interface, Direction, Sender, Receiver, and collection unit.
[FGR-BC0401_image_1.png]
The following table outlines the Counter-related items in the NSA system.
[FGR-BC0401_tables.md - Table 1]

The collected types of counters in each interface/direction are as follows:
 Packet loss count
 Packet out-of-sequence (OOS) count
 Packet count
 Packet loss rate
Note: F1-U path is not present in the integrated CU-DU shape. Hence, the counters for F1-U are not provided in this shape.

Stand Alone
There are two external networks of SA virtualization systems. One of them is the back-haul network between Core and gNB-UP, and is connected by the N3 interface. The other is mid-haul network between gNB-UP and DU, and is connected by the F1 interface. To measure network quality, a gNB collects the statistics for the number of lost packets and the number of out-of-sequence packets for user packets in the GTP layer. For this purpose, the sending side adds a sequence number for each GTP packet for transmission. The receiving side checks the sequence number for a received packet. If a packet with the specific sequence number is not received, it is judged as a lost packet. If there is a packet whose order is changed based on the sequence number, the out-of-sequence packet count is increased. The following figure depicts what statistics items CU-UP and DU can collect. The following table outlines Interface, Direction, Sender, Receiver, and Collection Unit.
The following figure depicts the SA architecture.
[FGR-BC0401_image_2.png]

The following table outlines Counter-related items in the SA system.
[FGR-BC0401_tables.md - Table 2]
The collected types of counters in each interface/direction are as follows:
 Packet loss count
 Packet out-of-sequence (OOS) count
 Packet count
 Packet loss rate
Note: F1-U path is not present in integrated CU-DU shape. Hence, the counters for F1-U are not provided in this shape.

#### When to Use
For basic call procedure, this feature requires to be operated by default.
#### Feature Optimization
GTP SN marking & Loss/OOS counting is configured by sequence-number-flag ( [AUPF] gtp-gw-info-entry ) only for S1-U / N3 interface.
For F1-U interface, GTP SN marking & Loss/OOS counting is always activated regardless of sequence-number-flag ([AUPF] gtp-gw-info-entry )
### SYSTEM OPERATION
This section describes how to configure the feature in Samsung system and provides associated key parameters, counters, and KPIs.
#### How to Activate
This section provides the information that you need to configure the feature.
Preconditions
There are no specific preconditions to active this feature.

Activation Procedure
To activate this feature, do the following:
 Run gtp-gw-info-entry and set sequence-number-flag to true. (Only for S1-U/N3 interface)
Note: For F1-U/Xn-U interface, GTP SN marking & Loss/OOS counting is always activated.

Deactivation Procedure
To deactivate this feature, do the following:
 Run gtp-gw-info-entry and set sequence-number-flag to false. (Only for S1-U/N3 interface)
#### Activation Confirmation
Verify if GTP Packet contains GTP sequence number when sequence-number-flag is enabled.
#### Key Parameters
This section describes the key parameters for activation, deactivation and configuration of the feature.
Activation/Deactivation
Parameters To activate or deactivate the feature, run the associated commands and set the key parameters.
Parameter Descriptions of gtp-gw-info-entry
[FGR-BC0401_tables.md - Table 3]

Configuration Parameters
To configure the feature settings, run the associated commands and set the key parameters.
Parameter Descriptions of gtp-peer-info/sgw-gtp-ip-entries
[FGR-BC0401_tables.md - Table 4]

Parameter Descriptions of gtp-peer-info/upf-gtp-ip-entries
[FGR-BC0401_tables.md - Table 5]

Parameter Descriptions of gtp-peer-info/enb-gtp-ip-entries
[FGR-BC0401_tables.md - Table 6]

Parameter Descriptions of gtp-peer-info/xn-gtp-ip-entries
[FGR-BC0401_tables.md - Table 7]

#### Counters and KPIs
The following table outlines the main counters associated with this feature.
[FGR-BC0401_tables.md - Table 8]
Note: The counter for ‘F1-U UL Interface per QCI’ is provided only for NSA operation.

[FGR-BC0401_tables.md - Table 9]
Note: The counter for ‘F1-U UL Interface collected in UP per QCI’ is provided only for NSA operation.
Note: The counter for the F1-U section is provided only for the CU-DU separation scenario.

[FGR-BC0401_tables.md - Table 10]
Note: The counter for ‘F1-U UL Interface per UPC’ is provided only for SA operation.
Note: The counter for ‘F1-U UL Interface per UPC’ might not be provided to the operator to which ‘F1-U, XN-U UL Interface collected in UPC per 5QI per SNSSAI’ is provided.

[FGR-BC0401_tables.md - Table 11]
Note: The counter ‘F1-U UL Interface collected in UP per UP’ is provided only for SA operation.
Note: The counter for ‘F1-U UL Interface collected in UP per UP’ might not be provided to the operator to which ‘F1-U, XN-U UL Interface collected in UPP per 5QI per SNSSAI’ is provided.
Note: The counter for the F1-U section is provided only for the CU-DU separation scenario.

[FGR-BC0401_tables.md - Table 12]
Note: The counter of ‘F1-U DL Interface per QCI’ and ‘F1-U DL Interface per PRC per QCI’ are provided only for NSA operation.
Note: The counter for F1-U section is provided only for CU-DU separation scenario.
Note: The counter of ‘F1-U DL Interface per QCI’ might not be provided to the operator to which ‘F1-U DL Interface per PRC per QCI’ is provided.

[FGR-BC0401_tables.md - Table 13]
Note: The counter of ‘F1-U DL Interface per DU’ and ‘F1-U DL Interface per PRC per DU’ provided only for SA operation.
Note: The counter for F1-U section is provided only for CU-DU separation scenario.

Note: The counter of ‘F1-U DL Interface per DU’ and ‘F1-U DL Interface per PRC per DU’ might not be provided to the operator to which ‘DL F1-U, Xn-U Interface per PRC per 5QI per S-NSSAI’ is provided.

[FGR-BC0401_tables.md - Table 14]
Note: Above counters are provided only for NSA for VNF CU

[FGR-BC0401_tables.md - Table 15]
Note: The above counters are provided only for NSA for CNF CU.

[FGR-BC0401_tables.md - Table 16]
Note: The above counters are provided only for SA for VNF CU.

[FGR-BC0401_tables.md - Table 17]
Note: The above counters are provided only for SA for CNF CU

[FGR-BC0401_tables.md - Table 18]
Note: Above counters are provided only for SA for CNF CU

[FGR-BC0401_tables.md - Table 19]
Note: The counter for ‘F1-U, XN-U UL Interface collected in UPC per 5QI per SNSSAI’ and ‘F1-U, XN-U UL Interface collected in UPP per 5QI per SNSSAI’ are provided only for SA operation.

[FGR-BC0401_tables.md - Table 20]
Note: The counter for ‘DL F1-U, Xn-U Interface per PRC per 5QI per S-NSSAI’ is provided only for SA operation.

 Per PRC counters: The PRC index indicates the location of the DPP in vDU where the GTP entity operates for each DRB in a multi-DPP structure.

[FGR-BC0401_tables.md - Table 21]
#### Troubleshooting
Verify that sequence-number-flag is set to true when this function does not operate normally. Repeat the activation/deactivation procedure described in How to Activate section.
If the problem persists after trying the above-mentioned actions, contact Samsung technical support.

### REFERENCE
None
