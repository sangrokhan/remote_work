/**
 * Week 01 contract types (contract-first baseline)
 * - language-agnostic fields are aligned to schema-contract.yaml and
 *   workflow-domain.schema.json
 */

export const WORKFLOW_ID_REGEX = /^[a-z][a-z0-9_-]{1,63}$/;
export const NODE_ID_REGEX = /^[a-z][a-z0-9_-]{1,63}$/;
export const STATE_KEY_REGEX = /^[a-z][a-zA-Z0-9_-]{0,127}$/;
export const VERSION_REGEX = /^\d+\.\d+\.\d+$/;

export type SchemaVersion = "1.0.0";

export interface WorkflowSchemaResponse {
  schemaVersion: SchemaVersion;
  workflowId: string;
  version: string;
  workflowName?: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface WorkflowNode {
  id: string;
  label: string;
  description: string;
  stateKey: string;
  metadata?: Record<string, unknown>;
  order: number;
}

export interface WorkflowEdge {
  id?: string;
  from: string;
  to: string;
  label?: string;
  metadata?: Record<string, unknown>;
}

export interface ErrorResponse {
  code: ErrorCode;
  message: string;
  requestId: string;
}

export type ErrorCode =
  | "WORKFLOW_NOT_FOUND"
  | "INVALID_WORKFLOW_ID"
  | "INVALID_WORKFLOW_PAYLOAD"
  | "WORKFLOW_REGISTRY_ERROR"
  | "INTERNAL_ERROR";
