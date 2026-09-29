# Terraform module for the protein-diffusion inference service.
# Applies the same Deployment + Service contract as ../k8s.yaml onto an
# existing Kubernetes cluster via the kubernetes provider.
#
#   terraform init && terraform apply -var image=<registry>/protein-diffusion:tag
#
# Requires a configured kubeconfig; not applied against a live cluster here.

terraform {
  required_version = ">= 1.5"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.32"
    }
  }
}

variable "image" {
  description = "Container image for the inference service"
  type        = string
}

variable "replicas" {
  description = "Desired pod count"
  type        = number
  default     = 2
}

variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "default"
}

variable "model_path" {
  description = "In-container path to the mounted DDPM checkpoint"
  type        = string
  default     = "data/processed/ddpm.pt"
}

variable "landscape_path" {
  description = "In-container path to the measured oracle parquet"
  type        = string
  default     = "data/processed/gb1.parquet"
}

locals {
  labels = { app = "protein-diffusion" }
}

resource "kubernetes_deployment" "serve" {
  metadata {
    name      = "protein-diffusion"
    namespace = var.namespace
    labels    = local.labels
  }
  spec {
    replicas = var.replicas
    selector { match_labels = local.labels }
    template {
      metadata { labels = local.labels }
      spec {
        container {
          name  = "serve"
          image = var.image
          args  = ["uvicorn", "protein_diffusion.serve:app",
                   "--host", "0.0.0.0", "--port", "8000"]
          port {
            container_port = 8000
            name           = "http"
          }
          env {
            name  = "MODEL_PATH"
            value = var.model_path
          }
          env {
            name  = "LANDSCAPE_PATH"
            value = var.landscape_path
          }
          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }
          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 15
            period_seconds        = 20
          }
          resources {
            requests = { cpu = "500m", memory = "1Gi" }
            limits   = { cpu = "2", memory = "4Gi" }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "serve" {
  metadata {
    name      = "protein-diffusion"
    namespace = var.namespace
    labels    = local.labels
  }
  spec {
    selector = local.labels
    port {
      port        = 80
      target_port = 8000
      name        = "http"
    }
    type = "ClusterIP"
  }
}

output "service_name" {
  value = kubernetes_service.serve.metadata[0].name
}
