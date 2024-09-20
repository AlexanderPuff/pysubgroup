import pandas as pd
try:
    import cudf
except ImportError:
    cudf = None

class DataFrameConfig:
    PANDAS = 'pandas'
    CUDF = 'cudf'
    
    _current_df_type = None

    @classmethod
    def set_dataframe_type(cls, df):
        if isinstance(df, pd.DataFrame):
            cls._current_df_type = cls.PANDAS
        elif cudf and isinstance(df, cudf.DataFrame):
            cls._current_df_type = cls.CUDF
        else:
            raise TypeError("Unsupported DataFrame type. Use pandas or cuDF DataFrame.")

    @classmethod
    def get_dataframe_type(cls):
        if cls._current_df_type is None:
            raise RuntimeError("DataFrame type not set. Call set_dataframe_type first.")
        return cls._current_df_type

    @classmethod
    def is_pandas(cls):
        return cls.get_dataframe_type() == cls.PANDAS

    @classmethod
    def is_cudf(cls):
        return cls.get_dataframe_type() == cls.CUDF
    
def ensure_df_type_set(func):
    def wrapper(df, *args, **kwargs):
        if DataFrameConfig._current_df_type is None:
            DataFrameConfig.set_dataframe_type(df)
        return func(df, *args, **kwargs)
    return wrapper
