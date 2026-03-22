.PHONY: build test lint format docker-up docker-down clean setup

# Source ROS2 before any colcon commands
ROS_SETUP := source /opt/ros/humble/setup.bash

build:
	bash -c "$(ROS_SETUP) && colcon build --symlink-install"

test: build
	bash -c "$(ROS_SETUP) && source install/setup.bash && colcon test && colcon test-result --verbose"

test-python:
	pytest tests/python/ -v --tb=short

lint:
	flake8 python/ dashboard/ transport/ tests/python/
	black --check --line-length 100 python/ dashboard/ transport/ tests/python/

format:
	black --line-length 100 python/ dashboard/ transport/ tests/python/

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

proto:
	protoc --python_out=transport/ transport/proto/*.proto

clean:
	rm -rf build/ install/ log/

setup:
	bash scripts/setup_env.sh
