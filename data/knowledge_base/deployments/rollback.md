---
title: Production Deployment Rollback Runbook
service: platform
environment: production
document_type: runbook
version: "1.0"
---

# Production Deployment Rollback Runbook

## Purpose

This runbook describes the standard procedure for rolling back a
production deployment when the current version causes service
degradation or failure.

## When to Roll Back

Consider rollback when:

- deployment introduces critical errors
- error rate increases significantly
- service becomes unavailable
- application startup fails
- database compatibility issues are detected
- a critical regression is confirmed

## Pre-Rollback Checks

Before initiating rollback:

1. Confirm the affected service.
2. Identify the current deployment version.
3. Identify the previous stable version.
4. Check active incidents.
5. Verify that rollback is technically safe.

## Rollback Procedure

First identify the deployment:

kubectl rollout history deployment/<deployment-name>

Check rollout status:

kubectl rollout status deployment/<deployment-name>

Rollback:

kubectl rollout undo deployment/<deployment-name>

Verify:

kubectl rollout status deployment/<deployment-name>

Then check:

- application logs
- pod status
- error rate
- service health

## Post-Rollback

Record:

- deployment ID
- failed version
- rollback version
- reason
- timestamp
- incident ID

The incident should remain open until service health is confirmed.

## Important

Rollback should not be performed automatically by an AI assistant
without explicit human approval.