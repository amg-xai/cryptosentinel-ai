# CryptoSentinel AI — Kubernetes Deployment

## Prerequisites
- kubectl configured
- minikube (local) or any K8s cluster
- Docker image pushed to GHCR

## Local deployment with minikube

```bash
# Start minikube
minikube start --memory=4096 --cpus=4

# Apply all manifests in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml      # Edit values first
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/statefulset-postgres.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/deployment-dashboard.yaml
kubectl apply -f k8s/hpa-api.yaml
kubectl apply -f k8s/network-policy.yaml

# Check status
kubectl get pods -n cryptosentinel
kubectl get services -n cryptosentinel

# Access dashboard
minikube service cryptosentinel-dashboard-service -n cryptosentinel
```

## Resource requirements
- API: 2 replicas × 250m CPU, 512Mi RAM (guaranteed)
- Dashboard: 1 replica × 250m CPU, 512Mi RAM
- PostgreSQL: 250m CPU, 256Mi RAM + 10Gi storage
- HPA: scales API from 2 to 10 replicas based on CPU/memory

## Production considerations
- Replace Secrets with Vault Agent sidecar injection
- Use cert-manager for TLS certificates
- Add Prometheus ServiceMonitor for K8s-native metrics scraping
- Use StatefulSet for Kafka with persistent storage
