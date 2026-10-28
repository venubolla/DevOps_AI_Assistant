---
title: Kubernetes CrashLoopBackOff Runbook
service: platform
environment: production
document_type: runbook
version: "1.0"
---

# Kubernetes CrashLoopBackOff Runbook

## Overview

CrashLoopBackOff occurs when a Kubernetes container repeatedly starts,
fails, and is restarted by Kubernetes.

The Kubernetes platform progressively increases the delay between
restart attempts.

## Common Causes

The most common causes include:

1. Application startup failure
2. Missing environment variables
3. Invalid application configuration
4. Missing secrets
5. Dependency connectivity failure
6. Insufficient CPU or memory
7. Incorrect container command
8. Application process exiting unexpectedly

## Troubleshooting Procedure

### Step 1 — Check Pod Status

Run:

kubectl get pods -n <namespace>

Look for pods showing:

CrashLoopBackOff

### Step 2 — Describe the Pod

Run:

kubectl describe pod <pod-name> -n <namespace>

Check:

- Events
- Container state
- Exit code
- Restart count
- Environment configuration

### Step 3 — Check Current Logs

Run:

kubectl logs <pod-name> -n <namespace>

Look for:

- startup exceptions
- configuration errors
- connection failures
- authentication failures

### Step 4 — Check Previous Container Logs

If the container has already restarted:

kubectl logs <pod-name> -n <namespace> --previous

This is particularly important because the current container may not
contain the logs from the failed process.

### Step 5 — Check Resource Configuration

Inspect CPU and memory requests and limits.

Run:

kubectl describe pod <pod-name> -n <namespace>

Look for:

- OOMKilled
- insufficient resources
- invalid resource configuration

## Recommended Resolution

Do not immediately restart the pod repeatedly.

First identify the underlying application or configuration failure.

If the failure was introduced by a deployment, compare the current
version with the previous stable version.

If the issue cannot be resolved quickly and the deployment is causing
production impact, follow the deployment rollback procedure.

## Escalation

Escalate to the application owner when:

- application startup repeatedly fails
- required secrets are unavailable
- database connectivity is failing
- the issue persists after configuration correction

## Related Runbooks

- Deployment Rollback
- High Memory Usage
- Service Unavailability