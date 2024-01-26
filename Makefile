start_labelstudio:
# https://hub.docker.com/r/heartexlabs/label-studio
	docker-compose up --build

kill_labelstudio:
	docker-compose down

convert_annotations:
	python convert_annotations.py --json_path "C:/Users/yannick.gibson/projects/school/BP/bachelors_thesis/myfiles/label-studio-exports/2024-01-25-21-11_min.json" --video_path "C:/Users/yannick.gibson/projects/work/important/ball-tracker/videos/ping_05_cam_2.mp4"