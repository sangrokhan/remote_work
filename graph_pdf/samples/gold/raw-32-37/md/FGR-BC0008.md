# Chapter 2 Bearer Control
## FGR-BC0008, DSCP Based Scheduling Adjustment
### INTRODUCTION
This feature provides DSCP based DL scheduling weight value for DL scheduling prioritization over non-GBR bearers and provides DSCP based service group mapping for UL pre-scheduling.
The gNB-CU transmits the DL packet outer IP DSCP value marked by the UPF to MCG path and SCG path.
The gNB-DU determines the DL scheduling weight according to the DSCP value for the DL proportional fair scheduling, and maps UL pre-scheduling group for each DSCP for UL pre-scheduling.
### BENEFIT
Users experience a better quality of service for latency and throughput for the specific service configured as high priority by the operator using the IP headers DSCP.
### RELEASE HISTORY
 SVR24A
Description
The feature is provided by new system (uADPF)
Enhancement
o Before Change: It does not operate in uADPF
o After Change: It operates in uADPF
 SVR24B
Description
The feature support DL scheduling based on inner IP DSCP value as well as outer IP DSCP value.
Enhancement
o Before Change: It only supports DL scheduling based on outer IP DSCP value.

o After Change: It supports DL scheduling based on outer or inner IP DSCP value and the operators could select to use which type DSCP value
 Newly added parameters: dscp-bypass-function container
### DEPENDENCY
 Required Network Elements: DU/RU vDU RAN(UP)
 Related Radio Technology: NR (5G)
 Prerequisite Features
o FGR-RS0703 QoS Scheduling
o FGR-RS7001 UL pre-scheduling
 Others: The core network should support DSCP marking in IP header for downlink data packets to enable this feature.
### LIMITATION
 DL scheduling adjustment is applied to only non-GBR bearers, not GBR bearers.
 The feature is provided by only systems that support FR1.
 For the MCG path of the EN-DC and NE-DC split bearer, the feature only defines the transmission of the outer IP DSCP of the X2 interface.
 The scheduling adjustment in the eNB cell for each DSCP value is supported by LTE-SW4213 and LTE-ME3319.
### SYSTEM IMPACT
This section describes how this feature impacts the network functions and capabilities.
Interdependencies between Features
This feature affects periodic UL grant activation (FGR-RS7001 UL pre-scheduling) and DL proportional fair scheduling metric for non-GBR bearers. (FGR-RS0703 QoS Scheduling).
If this feature is on in the gNB-CU, the gNB-CU does not mark the DSCP field of DL packets based on the 'FGR-CC4203 QoS Profile-based DSCP Mapping' feature. The gNB-CU keeps the DSCP value transmitted from the core network when sending DL packets to gNB-DU. The gNB-DU applies UL pre-scheduling and adjusts DL scheduling priority based on the DSCP of DL packets.
Performance and Capacity
None
Coverage
None

Interfaces
F1-U DL packet’s outer IP DSCP becomes the value when gNB is received from the core network.
### FEATURE DESCRIPTION
This feature provides DL scheduling weight value for DL scheduling prioritization over non-GBR bearers and provides periodic UL grant depending on the service.
To identify the service at gNB, the packet inspection should be performed at the core network according to operator’s network configuration. The core network marks specific DSCP value in outer IP header for the service configured by operator.
The gNB monitors DSCP value in outer IP header. The gNB applies periodic UL grant and DL scheduling weight adjustment for the corresponding bearer.
For the CA bearers, gNB-DU transmits DL packet outer IP DSCP value for both of Pcell and Scell.
The feature FGR-RS7001 defines the specific UL grant operation per UL pre-scheduling group.
If the dscp-index-ul ( ul-scheduling-dscp ) that is equal to DSCP value in DL Packet received from gNB-CU exists, ul-preschedule-group ( ul-scheduling-dscp ) is used to indicate latency-group-id ( latency-group ) of FGR-RS7001 feature at gNB-DU.
Additionally, If the dscp-index-dl ( dl-scheduling-dscp ) that is equal to DSCP value in DL Packet received from gNB-CU exists, the DL scheduling weight is determined by non-gbr-pf-weight-dscp-dl ( dl-scheduling-dscp ). This weight value is used for calculating DL proportional fair scheduling metric in the feature FGR-RS0703. As the weight value increases, the scheduling opportunity for the corresponding DSCP increases.
For the EN-DC split bearer, refer to LTE-SW4213 and LTE-ME3319 for scheduling adjustment in the eNB cell for each DSCP value.
The operator can configure dscp-bypass-mcg ( nr-dscp-bypass-entries ) and/or dscp-bypass-scg ( nr-dscp-bypass-entries ) whether to use outer IP DSCP value or inner IP DSCP. If set to 0, it means feature off to corresponding path. If set to 1, it uses outer IP DSCP value, and if set to 2, it uses inner IP DSCP value.
These Two parameters are set for each qos-5qi ( nr-dscp-bypass-entries ) and slice-index ( nr-dscp-bypass-entries ).
If the dscp-bypass-enable ( dscp-bypass-function ) is on and the dscp-bypass-mcg ( nr-dscp-bypass-entries ) and/or dscp-bypass-scg ( nr-dscp-bypass-entries ) are not set to 0, the gNB-CU forwards the DSCP value transmitted from the core network according to operator configuration when sending DL packets to gNB-DU.

For the EN-DC split bearer, the operator must configure dscp-bypass-mcg ( dscp-bypass-entries ) and/or dscp-bypass-scg ( dscp-bypass-entries ) for each qci( dscp-bypass-entries ). Setting method is the same as above parameters.
If the dscp-bypass-enable ( dscp-bypass-function ) is on and the dscp-bypass-mcg ( dscp-bypass-entries ) and/or dscp-bypass-scg ( dscp-bypass-entries ) are not set to 0, the gNB-CU forwards the DSCP value transmitted from the core network according to operator configuration when sending DL packets to gNB-DU and/or eNB.
#### When to Use
This feature is mainly used for increasing user throughput in congestion and for reducing packet latency. It is suggested to use this feature to provide differential service to the traffic classified by DSCP in core network.
#### Feature Optimization
None
### SYSTEM OPERATION
This section describes how to configure the feature in Samsung system and provides associated key parameters, counters, and KPIs.
#### How to Activate
This section provides the information that you need to configure the feature.
Preconditions
There are no specific preconditions to activate this feature.
Activation Procedure
To activate this feature, do the following:
 Run dscp-bypass-function and set dscp-bypass-enable to on, then run dscp-bypass-function/nr-dscp-bypass-entries and/or dscp-bypass-function/dscp-bypass-entries and set dscp-bypass-mcg and/or dscp-bypass-scg to a value other than 0.
 Run dl-scheduling-dscp and set non-gbr-pf-weight-dscp-dl to a value other than 0.
 Run ul-scheduling-dscp and set ul-preschedule-group to a value other than 255.

Deactivation Procedure
To deactivate this feature, do the following:
 Run dscp-bypass-function and set dscp-bypass-enable to off.
 Run dl-scheduling-dscp and set non-gbr-pf-weight-dscp-dl to 0.
 Run ul-scheduling-dscp and set ul-preschedule-group to 255.
#### Activation Confirmation
None
#### Key Parameters
This section describes the key parameters for activation, deactivation and configuration of the feature.
Activation/Deactivation Parameters
To activate or deactivate the feature, run the associated commands and set the key parameters.
Parameter Descriptions of dscp-bypass-function
[FGR-BC0008_tables.md - Table 1]

Parameter Descriptions of dscp-bypass-function/nr-dscp-bypass-entries
[FGR-BC0008_tables.md - Table 2]

Parameter Descriptions of dscp-bypass-function/dscp-bypass-entries
[FGR-BC0008_tables.md - Table 3]
Configuration Parameters
To configure the feature settings, run the associated commands and set the key parameters.
Parameter Descriptions of ul-scheduling-dscp
[FGR-BC0008_tables.md - Table 4]

Parameter Descriptions of dl-scheduling-dscp
[FGR-BC0008_tables.md - Table 5]
#### Counters and KPIs
There are no specific counters or Key Performance Indicators (KPIs) associated with this feature.
#### Troubleshooting
If the activation or deactivation process is not working properly, check whether the activation/deactivation parameter 'dscp-bypass-enable' is activated or not.
### REFERENCE
None