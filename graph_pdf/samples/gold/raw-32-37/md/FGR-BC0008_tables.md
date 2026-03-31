[FGR-BC0008_tables.md - Table 1]
| Parameter | Description |
| --- | --- |
| dscp-bypass-enable | This determines whether to enable or disable this feature:<br>off: This feature is not used.<br>on: This feature is used. |

[FGR-BC0008_tables.md - Table 2]
| Parameter | Description |
| --- | --- |
| slice-index | This leaf indicates the supported slice index. |
| qos-5qi | This leaf indicates a 5G QoS Identifier(5QI) used to map NR bearer traffic and DSCP. |
| dscp-bypass-mcg | This leaf indicates dscp-bypass function for DL packets on MCG leg.<br>0: off 1: It inserts outer IP DSCP value to DL packets.<br>2: It inserts inner IP DSCP value to DL packets |
| dscp-bypass-scg | This leaf indicates dscp-bypass function for DL packets on SCG leg.<br>0: off 1: It inserts outer IP DSCP value to DL packets.<br>2: It inserts inner IP DSCP value to DL packets |

[FGR-BC0008_tables.md - Table 3]
| Parameter | Description |
| --- | --- |
| qci | This leaf indicates a QoS Class Identifier(QCI). |
| dscp-bypass-mcg | This leaf indicates dscp-bypass function for DL packets on MCG leg.<br>0: off 1: It inserts outer IP DSCP value to DL packets 2: It inserts inner IP DSCP value to DL packets |
| dscp-bypass-scg | This leaf indicates dscp-bypass function for DL packets on SCG leg.<br>0: off 1: It inserts outer IP DSCP value to DL packets.<br>2: It inserts inner IP DSCP value to DL packets |

[FGR-BC0008_tables.md - Table 4]
| Parameter | Description |
| --- | --- |
| dscp-index-ul | This indicates the DSCP index used in the UL scheduler. |
| ul-preschedule-group | This indicates the latency group per DSCP for UL Prescheduling. |

[FGR-BC0008_tables.md - Table 5]
| Parameter | Description |
| --- | --- |
| dscp-index-dl | This indicates the DSCP index used in the DL scheduler. |
| non-gbr-pf-weight-dscp-dl | This indicates the relative weight index per DSCP in DL Proportional Fairness (PF) scheduler. As the value increases, the scheduling opportunity for the corresponding DSCP increases. |
