#!/bin/bash
echo
echo '1. COPYING LATEST TRAINED MODEL AND SCRIPTS'
echo

# if [[ -e ".env" ]]
#   then
#     # loading script parameters from .env
#     set -a            
#     source .env
#     set +a
# else
#     echo "No .env file with paramaters found. Exiting."
#     exit 1
# fi

echo
echo '2. BUILDING DOCKER IMAGE...'
echo
docker compose build

sleep 5

echo
echo '3. RUNNING DOCKER COMPOSE... prediction-service app will be available on port 5555'
echo
docker compose up &


