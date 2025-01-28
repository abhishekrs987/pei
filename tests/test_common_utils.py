import unittest
from unittest.mock import patch
from pyspark.sql import SparkSession
from utils.common_utils import (
    exception_handling,
    read_file_as_df,
    write_df_to_delta_table,
    clean_column_names,
    check_data_quality,
)
from pyspark.sql import functions as F
from pyspark.sql import DataFrame


class TestCommonUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Getting sparksession and preparing test data and test database"""
        self.spark = SparkSession.getActiveSession()
        orders = read_file_as_df(
            file_format="json",
            location="/Volumes/databricks_learning_01/default/test_data/Order.json",
            additional_options={"multiline": "true"},
        )
        self.orders_df = clean_column_names(orders)
        self.spark.sql("create database if not exists test_db;")

    def test_no_exception(self):
        """Test a function that does not raise an exception."""

        @exception_handling
        def func():
            return "No Exception"

        self.assertEqual(func(), "No Exception")

    def test_with_exception(self):
        """Test a function that raises an exception."""

        @exception_handling
        def func():
            raise ValueError("error occured in func()")

        with self.assertRaises(Exception) as context:
            func()

        self.assertIn("error occured in func()", str(context.exception))

    def test_clean_column_names(self):
        """Test clean column names to check if column names with special characters are getting removed and replaced with _"""
        data = [
            ("P001", "NY", "500.00"),
        ]
        df = self.spark.createDataFrame(
            data=data, schema="`product id` string,city string,`amount in` string"
        )
        expected_columns = ["product_id", "city", "amount_in"]
        cleaned_df = clean_column_names(df)
        self.assertEqual(cleaned_df.columns, expected_columns)

    def test_read_csv(self):
        """Test if file format csv is read properly with additional options provided"""
        file_format = "csv"
        location = "/Volumes/databricks_learning_01/default/test_data/Product.csv"
        additional_options = {"sep": ",", "header": "true"}
        products_df = read_file_as_df(file_format, location, additional_options)
        self.assertEqual(products_df.count(), 2)
        self.assertIsInstance(products_df, DataFrame)

    def test_read_json(self):
        """Test if file format json is read properly with additional options provided"""
        file_format = "json"
        location = "/Volumes/databricks_learning_01/default/test_data/Order.json"
        additional_options = {"multiline": "true"}
        orders_df = read_file_as_df(file_format, location, additional_options)
        self.assertEqual(orders_df.count(), 1)
        self.assertIsInstance(orders_df, DataFrame)

    def test_read_excel(self):
        """Test if file format excel is read properly with additional options provided"""
        file_format = "excel"
        location = "/Volumes/databricks_learning_01/default/test_data/Customer.xlsx"
        additional_options = {"sheet_name": "Sheet1"}
        customer_df = read_file_as_df(file_format, location, additional_options)
        self.assertEqual(customer_df.count(), 2)
        self.assertIsInstance(customer_df, DataFrame)

    def test_read_excel_invalid_additional_options(self):
        """Test if additional options provided is wrong an exception is caught"""
        file_format = "excel"
        location = "/Volumes/databricks_learning_01/default/test_data/Customer.xlsx"
        additional_options = {"wrong_key": "ok"}
        with self.assertRaises(Exception) as context:
            read_file_as_df(file_format, location, additional_options)
        self.assertTrue("unexpected keyword argument" in str(context.exception))

    def test_invalid_file_format(
        self,
    ):
        """Test if other than csv,json,excel file format is provided an exception is thrown"""
        file_format = "parquet"
        location = "path/to/file"
        additional_options = {}

        with self.assertRaises(Exception) as context:
            read_file_as_df(file_format, location, additional_options)

        self.assertTrue("Provide a valid file_format" in str(context.exception))

    @patch("pyspark.sql.SparkSession.getActiveSession", return_value=None)
    def test_no_spark_session(self, mock_get_active_session):
        """Test a scenario where spark session is not available"""
        file_format = "csv"
        location = "path/to/file"
        additional_options = {}

        with self.assertRaises(Exception) as context:
            read_file_as_df(file_format, location, additional_options)

        self.assertTrue("Unable to get SparkSession" in str(context.exception))

    def test_write_append(self):
        """Test if write operation executes properly in append mode"""
        database_name = "test_db"
        table_name = "test_table"
        load_type = "append"
        catalog_name = "databricks_learning_01"
        partition_columns = ["order_date"]

        result = write_df_to_delta_table(
            self.orders_df,
            catalog_name,
            database_name,
            table_name,
            load_type,
            partition_columns=partition_columns,
        )

        self.assertEqual(result, "success")

    def test_write_overwrite(self):
        """Test if write operation wroks properly with overwrite mode"""
        # Define table name components
        database_name = "test_db"
        table_name = "test_table"
        load_type = "overwrite"
        catalog_name = "databricks_learning_01"

        result = write_df_to_delta_table(
            self.orders_df,
            catalog_name,
            database_name,
            table_name,
            load_type,
        )

        self.assertEqual(result, "success")

    def test_write_upsert(self):
        """Test if write operation works properly with upsert mode"""
        # Define table name components
        database_name = "test_db"
        table_name = "test_table"
        load_type = "upsert"
        catalog_name = "databricks_learning_01"
        key_columns = ["order_id"]

        result = write_df_to_delta_table(
            self.orders_df,
            catalog_name,
            database_name,
            table_name,
            load_type,
            key_columns=key_columns,
        )

        self.assertEqual(result, "success")

    def test_write_upsert_keys_not_provided(self):
        """Test if upsert throws exception when key_columns are not provided"""
        # Define table name components
        database_name = "test_db"
        table_name = "test_table"
        load_type = "upsert"
        catalog_name = "databricks_learning_01"
        with self.assertRaises(Exception) as context:
            result = write_df_to_delta_table(
                self.orders_df,
                catalog_name,
                database_name,
                table_name,
                load_type,
            )

        self.assertTrue(
            "Provide key_columns to perform upsert operation" in str(context.exception)
        )

    def test_empty_dataframe(self):
        """Test empty dataframe write operation throws exception scenario"""
        df = self.orders_df.filter("order_id=0")

        with self.assertRaises(Exception) as context:
            write_df_to_delta_table(df,"test_catalog", "test_db", "test_table", "append")
        self.assertTrue(
            "Dataframe is Empty, Exiting Write Operation." in str(context.exception)
        )

    def test_incorrect_loadtype(self):
        """Test incorrect load_type value provided"""
        with self.assertRaises(Exception) as context:
            write_df_to_delta_table(
                self.orders_df,"test_catalog", "test_db", "test_table", "wrong_load_type"
            )
        print(context.exception)

        self.assertTrue("Provide right load_type value." in str(context.exception))

    def test_check_data_quality(self):
        df = self.orders_df.withColumn("value", F.lit(None))
        expected_condition = F.col("value").isNull()
        message = "Column value has null values"

        result_df = check_data_quality(df, expected_condition, message)

        # Check if '_data_quality' column exists
        self.assertIn("_data_quality", result_df.columns)

        # Check if the '_data_quality' column contains the expected message
        result = result_df.select("_data_quality").collect()[0][0]
        self.assertEqual(result["Column value has null values"], "true")

    @classmethod
    def tearDownClass(self):
        self.spark.sql("drop database if exists test_db cascade;")


if __name__ == "__main__":
    unittest.main()
