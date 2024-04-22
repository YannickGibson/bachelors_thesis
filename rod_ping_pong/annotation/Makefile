sl: start_labelstudio

start_labelstudio:
# https://hub.docker.com/r/heartexlabs/label-studio
	docker-compose up --build

kl: kill_labelstudio
kill_labelstudio:
	docker-compose down


#p:
#	ffmpeg -i "C:\Users\yannick.gibson\projects\work\important\ball-tracker\videos\blurred\2hrs.ts" -vf "trim=3600:3720, select=not(mod(n\,3))" -c:v libx264 -c:a aac -strict experimental output.mp4


convert_annotations:
#	ifndef CONFIG_FOLDER
#		$(error CONFIG_FOLDER is not set. Please provide the folder where JSON config is located.)
#	endif
	python ls2yolo.py \
	--json_path "C:\Users\yannick.gibson\projects\school\BP\bachelors_thesis\annotation\mydata\label_studio_exports\blurry-2024-02-19-10-24_min.json" \
	--video_path "C:\Users\yannick.gibson\projects\school\BP\bachelors_thesis\annotation\mydata\videos\blurred\scikit-video\2hrs_30fps_1to2min_0.83x_speed.mp4" \
	--output_base "mydata/ls2yolo_output/"