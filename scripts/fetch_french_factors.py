"""Download Kenneth French FF5 + Carhart momentum monthly factors (US).

Output: data/french_factors_monthly.parquet
  Columns: date (month-end), Mkt-RF, SMB, HML, RMW, CMA, UMD, RF
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "french_factors_monthly.parquet"

URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "umd": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}


def _download_csv_zip(url: str) -> str:
    r = httpx.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        return z.read(name).decode("latin-1")


def _parse_french_csv(text: str, value_cols: list[str]) -> pd.DataFrame:
    lines = text.splitlines()
    # Find header line
    hdr_idx = next(i for i, ln in enumerate(lines) if "," in ln and any(c in ln for c in value_cols))
    # Find end line (first empty line after data)
    end_idx = next((i for i in range(hdr_idx + 2, len(lines)) if not lines[i].strip()), len(lines))
    data = "\n".join(lines[hdr_idx:end_idx])
    df = pd.read_csv(io.StringIO(data))
    df.columns = [c.strip() for c in df.columns]
    # First column is YYYYMM, rename
    first = df.columns[0]
    df = df.rename(columns={first: "yyyymm"})
    df["yyyymm"] = df["yyyymm"].astype(str).str.strip()
    df = df[df["yyyymm"].str.match(r"^\d{6}$")]
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in value_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # French data in percent
    return df


def main() -> int:
    print("[french] downloading FF5 zip...")
    ff5_text = _download_csv_zip(URLS["ff5"])
    ff5_df = _parse_french_csv(ff5_text, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    print(f"  ff5 rows: {len(ff5_df)}, range: {ff5_df['date'].min()} .. {ff5_df['date'].max()}")

    print("[french] downloading UMD (momentum) zip...")
    umd_text = _download_csv_zip(URLS["umd"])
    # UMD column header may say "Mom" or "UMD" depending on year
    umd_df = _parse_french_csv(umd_text, ["Mom", "UMD"])
    if "Mom" in umd_df.columns and "UMD" not in umd_df.columns:
        umd_df = umd_df.rename(columns={"Mom": "UMD"})
    print(f"  umd rows: {len(umd_df)}, range: {umd_df['date'].min()} .. {umd_df['date'].max()}")

    merged = ff5_df.merge(umd_df[["date", "UMD"]], on="date", how="inner")
    keep = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]
    merged = merged[keep].sort_values("date").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"[french] wrote {OUT} -- rows={len(merged)}")
    print(merged.tail(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
