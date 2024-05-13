sl: start_labelstudio
start_labelstudio:
	docker-compose -f annotation/docker-compose.yaml up --build 

kl: kill_labelstudio
kill_labelstudio:
	docker-compose -f annotation/docker-compose.yaml down

convert_annotations: # Convert Label Studio annotations to YOLO format
	python annotaiton/ls2yolo.py \
	--json_path "$(json_path)" \
	--video_path "$(video_path)" \
	--output_base "$(output_base)"

train:
	python train.py \
	--data_path "$(data_path)" \
	--epochs $(epochs)
	--model_name "$(model_name)" \

play:
	python play_model.py