variable "REGISTRY" {
  default = "docker.io"
}

variable "REPO" {
  default = "dayuhub"
}

variable "TAG" {
  default = "v1.4"
}

variable "BASE_REPO" {
  default = "dayuhub"
}

variable "BASE_TAG" {
  default = "latest"
}

group "default" {
  targets = [
    "backend",
    "frontend",
    "datasource",
    "generator",
    "distributor",
    "controller",
    "scheduler",
    "monitor",
    "car-detection",
    "face-detection",
    "gender-classification",
    "age-classification",
    "model-switch-detection",
    "pedestrian-detection",
    "license-plate-recognition",
    "vehicle-detection",
    "exposure-identification",
    "category-identification",
    "traffic-detection",
    "road-context-segmentation",
    "traffic-signal-recognition",
    "vehicle-tracking",
    "vehicle-attribute-recognition",
    "vehicle-trajectory-prediction",
    "pedestrian-pose-estimation",
    "pedestrian-intent-recognition",
    "risk-graph-generation",
  ]
}

group "runtime" {
  targets = [
    "backend",
    "frontend",
    "datasource",
    "generator",
    "distributor",
    "controller",
    "scheduler",
    "monitor",
  ]
}

group "processors" {
  targets = [
    "car-detection",
    "face-detection",
    "gender-classification",
    "age-classification",
    "model-switch-detection",
    "pedestrian-detection",
    "license-plate-recognition",
    "vehicle-detection",
    "exposure-identification",
    "category-identification",
    "traffic-detection",
    "road-context-segmentation",
    "traffic-signal-recognition",
    "vehicle-tracking",
    "vehicle-attribute-recognition",
    "vehicle-trajectory-prediction",
    "pedestrian-pose-estimation",
    "pedestrian-intent-recognition",
    "risk-graph-generation",
  ]
}

group "dayubase" {
  targets = [
    "dayubase-default-amd64",
    "dayubase-default-arm64",
    "dayubase-jp4-amd64",
    "dayubase-jp4-arm64",
    "dayubase-jp5-amd64",
    "dayubase-jp5-arm64",
    "dayubase-jp6-amd64",
    "dayubase-jp6-arm64",
  ]
}

group "all-images" {
  targets = [
    "backend",
    "frontend",
    "datasource",
    "generator",
    "distributor",
    "controller",
    "scheduler",
    "monitor",
    "car-detection",
    "face-detection",
    "gender-classification",
    "age-classification",
    "model-switch-detection",
    "pedestrian-detection",
    "license-plate-recognition",
    "vehicle-detection",
    "exposure-identification",
    "category-identification",
    "traffic-detection",
    "road-context-segmentation",
    "traffic-signal-recognition",
    "vehicle-tracking",
    "vehicle-attribute-recognition",
    "vehicle-trajectory-prediction",
    "pedestrian-pose-estimation",
    "pedestrian-intent-recognition",
    "risk-graph-generation",
    "rtsp-server",
    "dayubase-default-amd64",
    "dayubase-default-arm64",
    "dayubase-jp4-amd64",
    "dayubase-jp4-arm64",
    "dayubase-jp5-amd64",
    "dayubase-jp5-arm64",
    "dayubase-jp6-amd64",
    "dayubase-jp6-arm64",
  ]
}

target "_image-common" {
  context = "."
  output = ["type=image,push=true"]
}

target "backend" {
  inherits = ["_image-common"]
  dockerfile = "build/backend.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/backend:${TAG}"]
}

target "frontend" {
  inherits = ["_image-common"]
  dockerfile = "build/frontend.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/frontend:${TAG}"]
}

target "datasource" {
  inherits = ["_image-common"]
  dockerfile = "build/datasource.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/datasource:${TAG}"]
}

target "generator" {
  inherits = ["_image-common"]
  dockerfile = "build/generator.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/generator:${TAG}"]
}

target "distributor" {
  inherits = ["_image-common"]
  dockerfile = "build/distributor.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/distributor:${TAG}"]
}

target "controller" {
  inherits = ["_image-common"]
  dockerfile = "build/controller.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/controller:${TAG}"]
}

target "scheduler" {
  inherits = ["_image-common"]
  dockerfile = "build/scheduler.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/scheduler:${TAG}"]
}

target "rtsp-server" {
  inherits = ["_image-common"]
  dockerfile = "build/rtsp_server.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/rtsp-server:${TAG}"]
}

target "monitor" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "monitor-${variant.name}"
  dockerfile = "build/monitor.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/monitor:${TAG}${variant.suffix}"]
}

target "car-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "car-detection-${variant.name}"
  dockerfile = "build/car_detection.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/car-detection:${TAG}${variant.suffix}"]
}

target "face-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "face-detection-${variant.name}"
  dockerfile = "build/face_detection.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/face-detection:${TAG}${variant.suffix}"]
}

target "gender-classification" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "gender-classification-${variant.name}"
  dockerfile = "build/gender_classification.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/gender-classification:${TAG}${variant.suffix}"]
}

target "age-classification" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "age-classification-${variant.name}"
  dockerfile = "build/age_classification.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/age-classification:${TAG}${variant.suffix}"]
}

target "model-switch-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "model-switch-detection-${variant.name}"
  dockerfile = "build/model_switch_detection.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/model-switch-detection:${TAG}${variant.suffix}"]
}

target "pedestrian-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "pedestrian-detection-${variant.name}"
  dockerfile = "build/pedestrian_detection.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/pedestrian-detection:${TAG}${variant.suffix}"]
}

target "license-plate-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "license-plate-recognition-${variant.name}"
  dockerfile = "build/license_plate_recognition.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/license-plate-recognition:${TAG}${variant.suffix}"]
}

target "vehicle-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "vehicle-detection-${variant.name}"
  dockerfile = "build/vehicle_detection.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-detection:${TAG}${variant.suffix}"]
}

target "exposure-identification" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "exposure-identification-${variant.name}"
  dockerfile = "build/exposure_identification.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/exposure-identification:${TAG}${variant.suffix}"]
}

target "category-identification" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "category-identification-${variant.name}"
  dockerfile = "build/category_identification.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/category-identification:${TAG}${variant.suffix}"]
}

target "traffic-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "traffic-detection-${variant.name}"
  dockerfile = "build/traffic_detection.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/traffic-detection:${TAG}${variant.suffix}"]
}

target "road-context-segmentation" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "road-context-segmentation-${variant.name}"
  dockerfile = "build/road_context_segmentation.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/road-context-segmentation:${TAG}${variant.suffix}"]
}

target "traffic-signal-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "traffic-signal-recognition-${variant.name}"
  dockerfile = "build/traffic_signal_recognition.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/traffic-signal-recognition:${TAG}${variant.suffix}"]
}

target "vehicle-tracking" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "vehicle-tracking-${variant.name}"
  dockerfile = "build/vehicle_tracking.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-tracking:${TAG}${variant.suffix}"]
}

target "vehicle-attribute-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "vehicle-attribute-recognition-${variant.name}"
  dockerfile = "build/vehicle_attribute_recognition.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-attribute-recognition:${TAG}${variant.suffix}"]
}

target "vehicle-trajectory-prediction" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "vehicle-trajectory-prediction-${variant.name}"
  dockerfile = "build/vehicle_trajectory_prediction.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-trajectory-prediction:${TAG}${variant.suffix}"]
}

target "pedestrian-pose-estimation" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "pedestrian-pose-estimation-${variant.name}"
  dockerfile = "build/pedestrian_pose_estimation.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/pedestrian-pose-estimation:${TAG}${variant.suffix}"]
}

target "pedestrian-intent-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "pedestrian-intent-recognition-${variant.name}"
  dockerfile = "build/pedestrian_intent_recognition.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/pedestrian-intent-recognition:${TAG}${variant.suffix}"]
}

target "risk-graph-generation" {
  inherits = ["_image-common"]
  matrix = {
    variant = [
      {
        name = "default"
        suffix = ""
      },
      {
        name = "jp4"
        suffix = "-jp4"
      },
      {
        name = "jp5"
        suffix = "-jp5"
      },
      {
        name = "jp6"
        suffix = "-jp6"
      },
    ]
  }
  name = "risk-graph-generation-${variant.name}"
  dockerfile = "build/risk_graph_generation.Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = "${BASE_TAG}${variant.suffix}"
  }
  tags = ["${REGISTRY}/${REPO}/risk-graph-generation:${TAG}${variant.suffix}"]
}

target "dayubase-default-amd64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_amd64.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-amd64"]
}

target "dayubase-default-arm64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_arm64.Dockerfile"
  platforms = ["linux/arm64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-arm64"]
}

target "dayubase-jp4-amd64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_amd64.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-jp4-amd64"]
}

target "dayubase-jp4-arm64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_jp4.Dockerfile"
  platforms = ["linux/arm64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-jp4-arm64"]
}

target "dayubase-jp5-amd64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_amd64.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-jp5-amd64"]
}

target "dayubase-jp5-arm64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_jp5.Dockerfile"
  platforms = ["linux/arm64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-jp5-arm64"]
}

target "dayubase-jp6-amd64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_amd64.Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-jp6-amd64"]
}

target "dayubase-jp6-arm64" {
  inherits = ["_image-common"]
  dockerfile = "build/dayubase_jp6.Dockerfile"
  platforms = ["linux/arm64"]
  args = {
    REG = REGISTRY
  }
  tags = ["${REGISTRY}/${REPO}/dayubase:${TAG}-jp6-arm64"]
}
