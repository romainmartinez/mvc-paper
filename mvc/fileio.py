from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def from_matlab_to_pandas(
        directory: Path, endswith: str, muscles: list, wide: bool = True, save: Path = None
) -> pd.DataFrame:
    data = []
    for ifile in directory.glob(f"*{endswith}.mat"):
        idataset = ifile.stem.replace(endswith, "").replace("MVE_Data_", "")
        mat = sio.loadmat(f"{ifile}")["MVE"]
        shape = mat.shape
        data.append(
            pd.DataFrame(
                [
                    dict(
                        participant=iparticipant,
                        dataset=idataset,
                        muscle=imuscle,
                        test=itest,
                        mvc=mat[iparticipant, imuscle, itest]
                        if len(shape) == 3
                        else np.nanmedian(mat[iparticipant, imuscle, itest])
                        if not np.isnan(mat[iparticipant, imuscle, itest]).all()
                        else np.nan,
                    )
                    for iparticipant in range(shape[0])
                    for imuscle in range(shape[1])
                    for itest in range(shape[2])
                ]
            ).dropna()
        )
    d = pd.concat(data).assign(
        dataset=lambda x: x["dataset"].str.replace("_", ""),
        # test=lambda x: (x["test"] + 1).astype(str).str.zfill(2),
        muscle=lambda x: np.array(muscles)[x["muscle"]],
    )
    if wide:
        d = d.pivot_table(
            index=["dataset", "participant", "muscle"],
            columns="test",
            values="mvc",
            fill_value=np.nan,
        ).reset_index()
    if save:
        d.to_csv(save, index=False)
    return d


if __name__ == "__main__":
    from constants import DATA_DIR, RAW_DATA_DIR, MUSCLES

    df_wide = from_matlab_to_pandas(
        directory=RAW_DATA_DIR,
        endswith="100_points",
        muscles=MUSCLES,
        wide=True,
        save=DATA_DIR / "df_wide.csv",
    )

    df_tidy = from_matlab_to_pandas(
        directory=RAW_DATA_DIR,
        endswith="100_points",
        muscles=MUSCLES,
        wide=False,
        save=DATA_DIR / "df_tidy.csv",
    )
