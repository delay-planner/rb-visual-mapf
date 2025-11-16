if [ -z "$1" ]; then
    echo "Usage: $0 <drone_id>"
    exit 1
fi

# drone_id from MAPF is 1-indexed, kirk_id is 0-indexed so we subtract 1
kirk_id=$(( $1 - 1 ))
port=$(( kirk_id + 8000 ))
tolerance=1.0

kirk run mission.rmpl -P scenario1 \
    --driver-command "curl -X POST -H 'Content-Type: application/json' -d '{\"start_cmd\":\"~A\", \"end_cmd\":\"~A\", \"kirk_id\": \"$kirk_id\"}' http://localhost:5000/submit" \
    --tolerance $tolerance -p $port --optimistic --verbose
