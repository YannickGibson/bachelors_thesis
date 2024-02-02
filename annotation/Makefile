sl: start_labelstudio

start_labelstudio:
# https://hub.docker.com/r/heartexlabs/label-studio
	docker-compose up --build

kl: kill_labelstudio
kill_labelstudio:
	docker-compose down

convert_annotations:
	ifndef CONFIG_FOLDER
		$(error CONFIG_FOLDER is not set. Please provide the folder where JSON config is located.)
	endif
	python ls2yolo.py \
	--json_path "C:/Users/yannick.gibson/projects/school/BP/bachelors_thesis/myfiles/label-studio-exports/2024-01-25-21-11_min.json" \
	--video_path "C:/Users/yannick.gibson/projects/work/important/ball-tracker/videos/ping_05_cam_2.mp4" \
	--output_path "C:/Users/yannick.gibson/projects/school/BP/bachelors_thesis/myfiles/label-studio-exports/2024-01-25-21-11_min.txt" \	x