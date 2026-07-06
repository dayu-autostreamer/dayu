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

locals {
  image_output = ["type=image,push=true"]
  multi_platform = ["linux/amd64", "linux/arm64"]

  runtime_targets = [
    "backend",
    "frontend",
    "datasource",
    "generator",
    "distributor",
    "controller",
    "scheduler",
    "monitor",
  ]

  processor_targets = [
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
    "traffic-object-detection",
    "road-context-segmentation",
    "traffic-signal-recognition",
    "vehicle-reidentification-tracking",
    "vehicle-attribute-recognition",
    "vehicle-trajectory-prediction",
    "pedestrian-cyclist-pose-estimation",
    "pedestrian-cyclist-intent-recognition",
    "traffic-risk-graph-inference",
  ]

  default_targets = concat(local.runtime_targets, local.processor_targets)

  jetpack_variants = [
    {
      name = "default"
      suffix = ""
      base_tag = BASE_TAG
      image_tag = TAG
    },
    {
      name = "jp4"
      suffix = "-jp4"
      base_tag = "${BASE_TAG}-jp4"
      image_tag = "${TAG}-jp4"
    },
    {
      name = "jp5"
      suffix = "-jp5"
      base_tag = "${BASE_TAG}-jp5"
      image_tag = "${TAG}-jp5"
    },
    {
      name = "jp6"
      suffix = "-jp6"
      base_tag = "${BASE_TAG}-jp6"
      image_tag = "${TAG}-jp6"
    },
  ]

  dayubase_targets = [
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

group "default" {
  targets = local.default_targets
}

group "runtime" {
  targets = local.runtime_targets
}

group "processors" {
  targets = local.processor_targets
}

group "dayubase" {
  targets = local.dayubase_targets
}

group "all-images" {
  targets = concat(local.default_targets, ["rtsp-server"], local.dayubase_targets)
}

target "_image-common" {
  context = "."
  output = local.image_output
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
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/datasource:${TAG}"]
}

target "generator" {
  inherits = ["_image-common"]
  dockerfile = "build/generator.Dockerfile"
  platforms = local.multi_platform
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
  platforms = local.multi_platform
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
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
  }
  tags = ["${REGISTRY}/${REPO}/rtsp-server:${TAG}"]
}

target "monitor" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "monitor-${variant.name}"
  dockerfile = "build/monitor.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/monitor:${variant.image_tag}"]
}

target "car-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "car-detection-${variant.name}"
  dockerfile = "build/car_detection.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/car-detection:${variant.image_tag}"]
}

target "face-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "face-detection-${variant.name}"
  dockerfile = "build/face_detection.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/face-detection:${variant.image_tag}"]
}

target "gender-classification" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "gender-classification-${variant.name}"
  dockerfile = "build/gender_classification.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/gender-classification:${variant.image_tag}"]
}

target "age-classification" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "age-classification-${variant.name}"
  dockerfile = "build/age_classification.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/age-classification:${variant.image_tag}"]
}

target "model-switch-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "model-switch-detection-${variant.name}"
  dockerfile = "build/model_switch_detection.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/model-switch-detection:${variant.image_tag}"]
}

target "pedestrian-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "pedestrian-detection-${variant.name}"
  dockerfile = "build/pedestrian_detection.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/pedestrian-detection:${variant.image_tag}"]
}

target "license-plate-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "license-plate-recognition-${variant.name}"
  dockerfile = "build/license_plate_recognition.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/license-plate-recognition:${variant.image_tag}"]
}

target "vehicle-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "vehicle-detection-${variant.name}"
  dockerfile = "build/vehicle_detection.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-detection:${variant.image_tag}"]
}

target "exposure-identification" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "exposure-identification-${variant.name}"
  dockerfile = "build/exposure_identification.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/exposure-identification:${variant.image_tag}"]
}

target "category-identification" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "category-identification-${variant.name}"
  dockerfile = "build/category_identification.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/category-identification:${variant.image_tag}"]
}

target "traffic-object-detection" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "traffic-object-detection-${variant.name}"
  dockerfile = "build/traffic_object_detection.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/traffic-object-detection:${variant.image_tag}"]
}

target "road-context-segmentation" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "road-context-segmentation-${variant.name}"
  dockerfile = "build/road_context_segmentation.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/road-context-segmentation:${variant.image_tag}"]
}

target "traffic-signal-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "traffic-signal-recognition-${variant.name}"
  dockerfile = "build/traffic_signal_recognition.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/traffic-signal-recognition:${variant.image_tag}"]
}

target "vehicle-reidentification-tracking" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "vehicle-reidentification-tracking-${variant.name}"
  dockerfile = "build/vehicle_reidentification_tracking.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-reidentification-tracking:${variant.image_tag}"]
}

target "vehicle-attribute-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "vehicle-attribute-recognition-${variant.name}"
  dockerfile = "build/vehicle_attribute_recognition.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-attribute-recognition:${variant.image_tag}"]
}

target "vehicle-trajectory-prediction" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "vehicle-trajectory-prediction-${variant.name}"
  dockerfile = "build/vehicle_trajectory_prediction.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/vehicle-trajectory-prediction:${variant.image_tag}"]
}

target "pedestrian-cyclist-pose-estimation" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "pedestrian-cyclist-pose-estimation-${variant.name}"
  dockerfile = "build/pedestrian_cyclist_pose_estimation.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/pedestrian-cyclist-pose-estimation:${variant.image_tag}"]
}

target "pedestrian-cyclist-intent-recognition" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "pedestrian-cyclist-intent-recognition-${variant.name}"
  dockerfile = "build/pedestrian_cyclist_intent_recognition.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/pedestrian-cyclist-intent-recognition:${variant.image_tag}"]
}

target "traffic-risk-graph-inference" {
  inherits = ["_image-common"]
  matrix = {
    variant = local.jetpack_variants
  }
  name = "traffic-risk-graph-inference-${variant.name}"
  dockerfile = "build/traffic_risk_graph_inference.Dockerfile"
  platforms = local.multi_platform
  args = {
    REG = REGISTRY
    BASE_REPO = BASE_REPO
    TAG = variant.base_tag
  }
  tags = ["${REGISTRY}/${REPO}/traffic-risk-graph-inference:${variant.image_tag}"]
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
