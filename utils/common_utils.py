import pyspark.pandas as ps
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from delta.tables import DeltaTable
import re
import traceback


def exception_handling(func):
    """The Python function's code is enclosed in a try/except.
    if there is any unhandled exception, returns error.

    Args:
        func: Python function.

    Returns:
        func output
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_log = {
                "error_message": f"Exception was caught in function '{func.__name__}',{str(e)}",
                "more details": traceback.format_exc(),
            }
            raise Exception(error_log)

    return wrapper


@exception_handling
def check_data_quality(
    df: DataFrame, expected_condition: str, message: str
) -> DataFrame:
    """
    Checks a given condition on the DataFrame and appends the result to a '_data_quality' column if True.

    Args:
        df (DataFrame): The input DataFrame.
        expected_condition (Column): The condition to check.
        message (str): The message to append to the '_data_quality' column if the condition is met.

    Returns:
        DataFrame: The DataFrame with the '_data_quality' column updated.

    Example condition and message
    expected_condition = F.col('B').isNull()
    message = "Column B has null values"

    Check data quality
    df = check_data_quality(df, expected_condition, message)
    df.show(truncate=False)
    """
    # Initialize '_data_quality' column if it doesn't exist
    if "_data_quality" not in df.columns:
        df = df.withColumn("_data_quality", F.create_map())

    # Update '_data_quality' column based on the condition
    df = df.withColumn(
        "_data_quality",
        F.when(
            expected_condition,
            F.map_concat(
                F.col("_data_quality"), F.create_map(F.lit(message), F.lit("true"))
            ),
        ).otherwise(F.col("_data_quality")),
    )

    return df


@exception_handling
def read_file_as_df(
    file_format: str.lower,
    location: str,
    additional_options: dict = {},
) -> DataFrame:
    """Read a file as Dataframe.

    Args:
        file_format (str): Supported file formats are csv,json,excel.
        location (str): absolute path to the file location.
        additional_options (dict, optional): additional options for spark.read or pandas read_excel, Defaults to {}.

    Returns:
        DataFrame: Spark Dataframe

    for additional options check out below,
        csv options: "https://spark.apache.org/docs/3.5.3/sql-data-sources-csv.html"
        json optins: "https://spark.apache.org/docs/3.5.3/sql-data-sources-json.html"
        excel options: "https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.read_excel.html"
    usage example:
    read_file(file_format = "csv",location = "dbfs:/path/to/dataset/",additional_options = {"sep":"|","header":"true"})

    """

    spark = SparkSession.getActiveSession()
    if not spark:
        raise ValueError("Unable to get SparkSession")

    if file_format in ["csv", "json"]:
        df = spark.read.format(file_format).options(**additional_options).load(location)

    elif file_format == "excel":
        psdf = ps.read_excel(location, **additional_options)
        df = psdf.to_spark()
    else:
        raise ValueError("Provide a valid file_format")
    return df


@exception_handling
def write_df_to_delta_table(
    df: DataFrame,
    catalog_name: str,
    database_name: str,
    table_name: str,
    load_type: str,  # append, overwrite, upsert
    key_columns: list = [],
    partition_columns: list = [],
    schema_evolution_mode: str = None,  # overwriteSchema, mergeSchema
) -> str:
    """Write a DataFrame as a delta table.

    Args:
        df (DataFrame): PySpark DataFrame to write.
        catalog_name (str, optional): Name of the catalog.
        database_name (str): Name of the database/schema
        table_name (str): Name of the table
        load_type (str): Way to load (append,overwrite,upsert)
        key_columns (list, optional): Key columns for performing upsert operation. Defaults to [].
        partition_columns (list, optional): Partition columns for physical partitioning of the table. Defaults to [].
        schema_evolution_mode (str, optional): way to handle schema changes. Defaults to None.


    Returns:
        "success"
    """
    if load_type not in ("append", "overwrite", "upsert"):
        raise ValueError("Provide right load_type value.")
    
    table_name = f"{catalog_name}.{database_name}.{table_name}"
    spark = SparkSession.getActiveSession()
    if not spark:
        raise ValueError("Unable to get SparkSession")

    # Use optimized writes to reduce the number of small files
    spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
    spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

    df = df.withColumn("_update_gmt_ts", F.current_timestamp())

    if df.count() == 0:
        raise ValueError("Dataframe is Empty, Exiting Write Operation.")

    df_writer = df.write.format("delta")
    if schema_evolution_mode:
        df_writer = df_writer.option(schema_evolution_mode, "true")
    if partition_columns:
        df_writer = df_writer.partitionBy(partition_columns)
    if not spark.catalog.tableExists(table_name):
        print("Delta table does not exists yet, creating a new delta table.")
        df_writer.saveAsTable(table_name)

    elif load_type == "overwrite" or load_type == "append":
        df_writer.mode(load_type).saveAsTable(table_name)
    elif load_type == "upsert":
        if key_columns:
            merge_condition_parts = [
                f"source.`{col}` <=> target.`{col}`" for col in key_columns
            ]
            merge_condition = " AND ".join(merge_condition_parts)
            delta_table = DeltaTable.forName(spark, table_name)
            (
                delta_table.alias("target")
                .merge(df.alias("source"), merge_condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            raise ValueError("Provide key_columns to perform upsert operation")

    return "success"


@exception_handling
def clean_column_names(df: DataFrame) -> DataFrame:
    """clean column names and use lowercase"""
    for c in df.columns:
        df = df.withColumnRenamed(c, re.sub(r"[^0-9a-zA-Z]", "_", c).lower())
    return df
