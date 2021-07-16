# run application using docker compose

if [ $1 == "base" ]; then
    docker-compose up --build stylegan_server ui
elif [ $1 == "pose" ]; then
    docker-compose up --build pose_server
else
    docker-compose up --build 
fi
