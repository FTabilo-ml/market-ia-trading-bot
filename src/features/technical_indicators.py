import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True)
    # indicador simple de cruce de medias
    df["golden_cross"] = (df["trend_ema_fast"] > df["trend_ema_slow"]).astype(int)
    return df
