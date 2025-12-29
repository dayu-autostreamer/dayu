export REG=repo:5000
docker pull "$REG"/dayuhub/monitor:v1.2-jp6
docker tag "$REG"/dayuhub/monitor:v1.2-jp6 "$REG"/dayuhub/monitor:v1.2
docker pull "$REG"/dayuhub/pedestrian-detection:v1.2-jp6
docker tag "$REG"/dayuhub/pedestrian-detection:v1.2-jp6 "$REG"/dayuhub/pedestrian-detection:v1.2
docker pull "$REG"/dayuhub/license-plate-recognition:v1.2-jp6
docker tag "$REG"/dayuhub/license-plate-recognition:v1.2-jp6 "$REG"/dayuhub/license-plate-recognition:v1.2
docker pull "$REG"/dayuhub/vehicle-detection:v1.2-jp6
docker tag "$REG"/dayuhub/vehicle-detection:v1.2-jp6 "$REG"/dayuhub/vehicle-detection:v1.2
docker pull "$REG"/dayuhub/exposure-identification:v1.2-jp6
docker tag "$REG"/dayuhub/exposure-identification:v1.2-jp6 "$REG"/dayuhub/exposure-identification:v1.2
docker pull "$REG"/dayuhub/category-identification:v1.2-jp6
docker tag "$REG"/dayuhub/category-identification:v1.2-jp6 "$REG"/dayuhub/category-identification:v1.2
