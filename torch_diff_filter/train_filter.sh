#!/usr/bin/env bash
source conf.sh


if [[ -z $IMAGE_TAG ]]
then
  echo "No docker tag provided. Cannot run docker image."
else
  echo "Warning: Removing containers with the prefix $CONTAINER_NAME* "
  docker rm -f $CONTAINER_NAME

docker run --gpus all \
			-it \
			--env-file conf.sh \
			--name $CONTAINER_NAME \
			-v /data/xiao/diff_filter/KITTI:/tf \
			-v /data/xiao/dataset:/tf/dataset \
			radiusaiinc/diffkalman:$IMAGE_TAG \
			/bin/bash -c "python3 run_PF_filter.py"
fi