#!/usr/bin/env bash

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm repo update

helm install promstack prometheus-community/kube-prometheus-stack
kubectl apply -f deployment.yml

kubectl port-forward svc/promstack-kube-prometheus-prometheus 9090:9090 &

kubectl port-forward svc/promstack-grafana 3000:80 &
