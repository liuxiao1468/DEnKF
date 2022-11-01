#!/usr/bin/env bash
source conf.sh


if [[ -z $IMAGE_TAG ]]
then
  echo "No docker tag provided. Cannot run docker image."
else
  echo "Warning: Removing containers with the prefix $CONTAINER_NAME* "
  docker rm -f $CONTAINER_NAME "$CONTAINER_NAME-tensorboard"

docker run --gpus all \
			-d \
			--env-file conf.sh \
			--name $CONTAINER_NAME \
			-v /data/xiao/diff_filter/KITTI:/tf \
			-v /data/xiao/dataset:/tf/dataset \
			radiusaiinc/diffkalman:$IMAGE_TAG \
			/bin/bash -c "python3 run_high_filter.py"

logdir="/tf/experiments/loss/high_v1.0"
docker run --name "$CONTAINER_NAME-tensorboard" \
			--gpus=all \
			--network host \
			-d \
			--env-file conf.sh \
			-v /data/xiao/diff_filter/KITTI:/tf \
			-v /data/xiao/dataset:/tf/dataset \
			radiusaiinc/diffkalman:$IMAGE_TAG \
			/bin/bash -c "tensorboard --logdir $logdir --host 0.0.0.0 --port 8093  --reload_multifile=true"
fi