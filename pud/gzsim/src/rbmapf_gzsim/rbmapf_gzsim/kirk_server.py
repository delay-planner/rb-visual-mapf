import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()


class RequestData(BaseModel):
    start_cmd: str
    end_cmd: str
    kirk_id: str


values = {}


@app.get("/")
async def always_true(key: str):
    seen = values.get("start", False)
    return JSONResponse(content={"result": seen})


@app.post("/submit")
async def submit_data(data: RequestData):
    global values
    response_data = {
        "start_cmd": data.start_cmd,
        "end_cmd": data.end_cmd,
        "kirk_id": data.kirk_id,
    }
    values["start"] = True
    return JSONResponse(content=response_data)


def main():
    uvicorn.run("kirk_server:app", host="127.0.0.1", port=5000, reload=True)


if __name__ == "__main__":
    main()
