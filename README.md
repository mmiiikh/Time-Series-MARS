# Time-Series-MARS
Repository for code base of time series model to forecast sales of MARS.

At this stage, the repository contains only the project structure and
Docker environment required to run the code in an isolated container.
.
1) Dockerfile
2) README.md
3) requirements.txt
4) src
   
    4.1.) main.py

## Requirements

- Docker

## Build Docker image

```bash
docker build -t time-series-mars .

## Run Container

```bash
docker run --rm time-series-mars

Expected Outcome: Time-Series-MARS project is running in Docker

## Notes
For checkpoint 1 there are no model implemented yet.
This checkpoint focuses on setting up the Docker environment.


