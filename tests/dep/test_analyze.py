import json
from x_ray.worker.tasks import analyze

if __name__ == "__main__":
    process_uid = "a_good_process_uid"
    with open("_ex_pt.geojson") as f:
        gj = json.load(f)
    gj_str = json.dumps(gj)
    start = "2023_01"
    stop = "2023_05"
    signed_url: str = analyze(start, stop, gj_str, process_uid)
    print(signed_url)