from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import sys
from . import common_utils


@common_utils.exception_handling
def aggregate_orders(
    orders: DataFrame,
) -> DataFrame:
    """
    Aggregates the orders by year,category,sub_category,customer_id.

    Args:
    orders (DataFrame): The input orders DataFrame.

    Returns:
    DataFrame: The aggregated DataFrame.
    """

    orders = orders.withColumn(
        "year", F.year(F.to_date(F.col("order_date"), format="d/M/yyyy"))
    )
    aggregated_orders = orders.groupBy(
        ["year", "category", "sub_category", "customer_id"]
    ).agg(F.sum("profit").alias("profit"))

    return aggregated_orders


@common_utils.exception_handling
def enrich_orders(
    orders: DataFrame,
    products: DataFrame,
    customers: DataFrame,
) -> DataFrame:
    """
    Enriches the orders DataFrame by joining it with the customers and products DataFrames.
    The function adds customer details (name, country) and product details (category, sub-category).
    It also rounds the profit column to 2 decimal places also brodcasts products and customers dataframes.
    to disabled broadcast join set spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    Args:
    orders (DataFrame): The orders DataFrame.
    products (DataFrame): The products DataFrame.
    customers (DataFrame): The customers DataFrame.

    Returns:
    DataFrame: The enriched orders DataFrame.
    """

    # Join orders with customers and products DataFrames
    orders_enriched = orders.join(
        F.broadcast(customers.select("customer_id", "customer_name", "country")),
        on="customer_id",
        how="left",
    ).join(
        F.broadcast(products.select("product_id", "category", "sub_category")),
        on="product_id",
        how="left",
    )

    # Round the profit column to 2 decimal places
    orders_enriched = orders_enriched.withColumn("profit", F.round(F.col("profit"), 2))

    return orders_enriched


@common_utils.exception_handling
def enrich_products(products: DataFrame) -> DataFrame:
    """
    Enriches the products DataFrame by removing duplicate entries based on the product_id column.

    Args:
    products (DataFrame): The products DataFrame.

    Returns:
    DataFrame: The enriched products DataFrame with duplicates removed.
    """
    # Remove duplicates based on the product_id column
    products_enriched = products.dropDuplicates(["product_id"])

    return products_enriched


@common_utils.exception_handling
def enrich_customers(customers: DataFrame) -> DataFrame:
    """
    Enriches the customers DataFrame by performing the following operations:
    1. Removes duplicate entries based on the customer_id column.
    2. Cleans and trims the customer names to remove non-alphabetic characters and extra spaces.
    3. Cleans and formats phone numbers to the US format and also fills Null if phone column doesn't align to US format.

    Args:
    customers (DataFrame): The customers DataFrame.

    Returns:
    DataFrame: The enriched customers DataFrame.
    """
    # Remove duplicates based on the customer_id column
    customers = customers.dropDuplicates(["customer_id"])

    # Clean and trim the customer names
    customers = customers.withColumn(
        "customer_name",
        F.trim(F.regexp_replace(F.col("customer_name"), r"[^a-zA-Z ]", "")),
    ).withColumn(
        "customer_name",
        F.trim(F.regexp_replace(F.col("customer_name"), r"\s{2,}", " ")),
    )

    # Clean and format phone numbers to the US format
    customers = customers.withColumn(
        "phone", F.trim(F.regexp_replace(F.col("phone"), r"[^0-9]", ""))
    )
    customers = customers.withColumn(
        "phone",
        F.concat(
            F.lit("+1 ("),
            F.substring(F.col("phone"), 1, 3),
            F.lit(") "),
            F.substring(F.col("phone"), 4, 3),
            F.lit("-"),
            F.substring(F.col("phone"), 7, 4),
        ),
    )
    customers = customers.withColumn(
        "phone",
        F.when(F.length(F.col("phone")) < 17, F.lit(None)).otherwise(F.col("phone")),
    )

    return customers
