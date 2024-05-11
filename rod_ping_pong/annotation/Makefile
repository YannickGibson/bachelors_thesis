sl: start_labelstudio
start_labelstudio:
	docker-compose up --build

kl: kill_labelstudio
kill_labelstudio:
	docker-compose down

convert_annotations: # Convert Label Studio annotations to YOLO format
	python ls2yolo.py \
	--json_path "$(json_path)" \
	--video_path "$(video_path)" \
	--output_base "$(output_base)"