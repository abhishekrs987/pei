import unittest
from pyspark.sql import SparkSession
from utils.transform_utils import (
    aggregate_orders,
    enrich_customers,
    enrich_orders,
    enrich_products,
)
from utils.common_utils import read_file_as_df, clean_column_names
from pyspark.sql import functions as F


class TestCommonUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.spark = SparkSession.getActiveSession()
        orders = read_file_as_df(
            file_format="json",
            location="/Volumes/databricks_learning_01/default/test_data/Order.json",
            additional_options={"multiline": "true"},
        )
        self.orders = clean_column_names(orders)

        products = read_file_as_df(
            file_format="csv",
            location="/Volumes/databricks_learning_01/default/test_data/Product.csv",
            additional_options={
                "sep": ",",
                "header": "true",
                "quote": '"',
                "escape": '"',
            },
        )
        self.products = clean_column_names(products)

        customers = read_file_as_df(
            file_format="excel",
            location="/Volumes/databricks_learning_01/default/test_data/Customer.xlsx",
        )
        self.customers = clean_column_names(customers)

        self.spark.sql("create database if not exists test_db;")

    def test_enrich_customers(self):
        """Test if enrich customers method is working as per the expectations."""
        result_df = enrich_customers(self.customers)
        self.assertEqual(
            result_df.filter(F.col("customer_id") == "PW-19240").count(), 1
        )
        self.assertEqual(
            result_df.filter(F.col("customer_id") == "PW-19240")
            .select("customer_name")
            .collect()[0][0],
            "Pierre Wener",
        )
        self.assertEqual(
            result_df.filter(F.col("customer_id") == "PW-19240")
            .select("phone")
            .collect()[0][0],
            "+1 (421) 580-0902",
        )

    def test_enrich_products(self):
        """Test if enrich products is working as per the expectation."""
        result_df = enrich_products(self.products)
        self.assertEqual(
            result_df.filter(F.col("product_id") == "FUR-CH-10002961").count(), 1
        )

    def test_enrich_orders(self):
        """Test if enrich orders is working as per the expectation"""
        result_df = enrich_orders(
            self.orders,
            enrich_products(self.products),
            enrich_customers(self.customers),
        )
        self.assertEqual(
            result_df.filter(F.col("order_id") == "CA-2016-122581")
            .select("customer_name")
            .collect()[0][0],
            "Pierre Wener",
        )
        self.assertEqual(
            result_df.filter(F.col("order_id") == "CA-2016-122581")
            .select("category")
            .collect()[0][0],
            "Furniture",
        )
        self.assertEqual(
            result_df.filter(F.col("order_id") == "CA-2016-122581")
            .select("profit")
            .collect()[0][0],
            63.69,
        )

    def test_aggregate_orders(self):
        """Test if aggreagte orders is working as per the expectation"""
        result_df = aggregate_orders(
            enrich_orders(
                self.orders,
                enrich_products(self.products),
                enrich_customers(self.customers),
            )
        )
        self.assertEqual(
            result_df.filter(F.col("customer_id") == "PW-19240")
            .select("profit")
            .collect()[0][0],
            63.69,
        )

    @classmethod
    def tearDownClass(self):
        self.spark.sql("drop database if exists test_db cascade;")


if __name__ == "__main__":
    unittest.main()
