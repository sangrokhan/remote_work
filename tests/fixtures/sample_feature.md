# Feature: Carrier Aggregation (CA)

**Feature ID**: feature:CA
**Generation**: both
**Category**: radio_resource_management

## Description

Carrier Aggregation enables a UE to receive and transmit on multiple component carriers simultaneously, increasing peak data rates by combining bandwidth from multiple carriers.

## KPI Impact

| KPI ID | KPI Name | Direction | Magnitude | Condition |
|--------|----------|-----------|-----------|-----------|
| kpi:dl_throughput | DL Throughput | + | high | multi-band capable UE |
| kpi:ul_throughput | UL Throughput | + | medium | UL CA capable UE |

## Controlling Parameters

| Parameter ID | Effect When Increased |
|-------------|----------------------|
| param:maxCaBands | More bands aggregated, higher peak throughput |
| param:caBandwidth | Wider bandwidth per component carrier |

## Feature Dependencies

| Feature ID | Dependency Type |
|------------|----------------|
| feature:MIMO | enables |
