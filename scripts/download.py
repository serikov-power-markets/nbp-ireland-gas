"""Download the open data of the project: National Gas Transmission REST API (Moffat physical flow, nominations, entry capacity
auction results) and the ONS daily System Average Price (SAP) of gas. No account needed."""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
API = "https://api.nationalgas.com/operationaldata/v1/publications/gasday"
PUBS = {"PUBOB2039": "moffat_physical_flow_mscm", "PUBOBJ1093": "moffat_nomination_exit_kwh", "PUBOBJ1596": "moffat_nomination_entry_kwh"}


def fetch(pid, start, end):
    rows = []
    for y in range(start, end + 1):
        for half in ((f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")):
            for attempt in range(8):
                r = requests.post(API, json={"fromDate": half[0], "toDate": half[1], "publicationIds": [pid], "latestValue": "Y"}, timeout=120)
                if r.status_code != 429:
                    break
                time.sleep(20 * (attempt + 1))
            r.raise_for_status()
            time.sleep(3)
            for blk in r.json():
                for p in blk["publications"]:
                    rows.append((p["applicableFor"], float(p["value"]) if p["value"] not in ("", None) else None))
    return pd.Series(dict(rows), name=PUBS[pid]).sort_index()


if __name__ == "__main__":
    end = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    frames = []
    for pid in PUBS:
        s = fetch(pid, 2018, end)
        print(pid, PUBS[pid], len(s), s.index.min() if len(s) else None, s.index.max() if len(s) else None, flush=True)
        frames.append(s)
    pd.concat(frames, axis=1).to_csv(DATA / "national_gas_moffat.csv", index_label="gas_day")
    sap_url = "https://www.ons.gov.uk" + re.search(r'href="(/file\?uri=[^"]*systemaveragepriceofgasdataset\d+\.xlsx)"',
                                                  requests.get("https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas", timeout=60).text).group(1)
    (DATA / "ons_sap.xlsx").write_bytes(requests.get(sap_url, timeout=120).content)
    print("SAP", sap_url)
    fxt = requests.get("https://data-api.ecb.europa.eu/service/data/EXR/D.GBP.EUR.SP00.A?format=csvdata&startPeriod=2018-01-01&detail=dataonly", timeout=60).text
    import io
    f = pd.read_csv(io.StringIO(fxt))
    (1 / f.set_index(pd.to_datetime(f.TIME_PERIOD)).OBS_VALUE).rename("eur_per_gbp").to_csv(DATA / "ecb_eurgbp.csv")
