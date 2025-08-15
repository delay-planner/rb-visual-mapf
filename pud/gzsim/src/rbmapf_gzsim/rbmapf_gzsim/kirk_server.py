import json
import uvicorn
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# List of queue of events seen from Kirk for each drone
sync = False
land = False
sync_name = ""
sync_seen = set()
values = [[]]
visited_set = set()

app = FastAPI()


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


class RequestData(BaseModel):
    start_cmd: str
    end_cmd: str
    kirk_id: int


class MissionFinishData(BaseModel):
    drone_id: int
    mission_name: str


class SyncFinishData(BaseModel):
    sync_name: str


def send_kirk_ack(event_id):
    """
    Send an event ack to kirk
    """

    port = 8000
    url = f"http://localhost:{port}/"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "event-ids": [event_id]
    }
    try:
        _ = requests.post(url, headers=headers, data=json.dumps(data))
    except Exception as e:
        print("Error with kirk ack", e)


@app.post("/done")
async def mission_finishes(data: MissionFinishData):
    drone_id = int(data.drone_id) - 1
    # print("received mission finish ack", data.drone_id, data.mission_name)
    # print(values)
    try:
        values[drone_id].remove(data.mission_name)
        send_kirk_ack(data.mission_name)
        # print("successfully removed")

        if "START" in data.mission_name and "LAND" in data.mission_name:
            global land
            land = True

        return JSONResponse(content={"success": True})
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"success": False})


@app.post("/done_sync")
async def sync_finishes(data: SyncFinishData):
    """
    Sends ack to Kirk that the sync finishes
    """
    send_kirk_ack(data.sync_name)
    # print("successfully removed")
    return JSONResponse(content={"success": True})


@app.get("/plan")
async def plan_new_mission():
    """
    Returns whether we're ready to plan a new mission!
    """
    global sync, sync_name

    if sync:
        content = {"sync_ready": True, "sync_name": sync_name}
        sync = False
        sync_name = ""
        return JSONResponse(content=content)
    else:
        return JSONResponse(content={"sync_ready": False, "sync_name": ""})


@app.get("/")
async def most_recent_mission(drone_id: int):
    """
    Input: drone_id
    Output:
    {
        mission_ready: bool,  # whether or not there is a mission ready
        mission_name: str,    # name of the mission to execute
        land: bool,           # if true, just land
    }
    """
    drone_id = int(drone_id) - 1
    # print(values)

    # Verify the drone id is valid
    if drone_id >= len(values):
        # print("unknown drone id!", drone_id, ">=", len(values))
        return JSONResponse(content={"result": False})

    # Check if the value exists, otherwise return False
    if len(values[drone_id]) > 0:
        # print("return mission")
        # print(values[drone_id][0])
        # There exists a mission to be returned, return the first
        return JSONResponse(content={"mission_ready": True, "mission_name": values[drone_id][0], "land": land})
    else:
        # print("nomission")
        # There are no missions at the moment
        return JSONResponse(content={"mission_ready": False, "mission_name": "", "land": land})


@app.post("/submit")
async def receive_kirk_event(data: RequestData):
    """
    Recieves an event from Kirk
    Each event has a start id and an end id and will look a like this:

    {
        "start_cmd": "+EVENT-NAME::START+",
        "end_cmd": "+EVENT-NAME::END+",
        "kirk_id": 0,
    }

    The kirk_id corresponds to from which kirk did we get the start event (the drone id)
    """
    global values, sync_seen, sync, sync_name, land, visited_set

    # Make the info a dictionary
    response_data = {
        "kirk_id": data.kirk_id,
        "end_cmd": data.end_cmd,
        "start_cmd": data.start_cmd,
    }

    # print(response_data)

    if "SYNC" in data.start_cmd:
        # It's a sync command! we'll want to update the global sync structures if it's actually a new command (not a duplicate from multiple drones)

        if data.end_cmd in sync_seen:
            return JSONResponse(content=response_data)
        else:
            sync_seen.add(data.end_cmd)
            sync = True
            sync_name = data.end_cmd

        return JSONResponse(content=response_data)
    elif "NOOP" in data.start_cmd:
        return JSONResponse(content=response_data)
    else:
        data.kirk_id = int(data.kirk_id)

        # Verify the kirk id is valid
        if data.kirk_id >= len(values):
            # print("unknown kirk id!", data.kirk_id)
            assert False

        if "LAND" in data.start_cmd:
            land = True

        # print("got msg from kirk")
        if (data.end_cmd, data.kirk_id) not in visited_set:
            visited_set.add((data.end_cmd, data.kirk_id))
            values[data.kirk_id].append(data.end_cmd)

        return JSONResponse(content=response_data)


def main():
    # global values

    # print("value len", len(values))
    uvicorn.run("kirk_server:app", host="127.0.0.1", port=5000, reload=True)


if __name__ == "__main__":
    main()
