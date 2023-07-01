from visibility.constants.database import TABLE_NAME
from visibility.logger import logging
from visibility.constants.training_pipeline import RAW_FILE_NAME
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
import os
import sys

import pandas as pd
import csv
import ast
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
MYSQL_URL = os.getenv('MYSQL_URL')
class ClimateData:
    def __init__(self):
        try:
            logging.info("Establish the connection")
            self.engine = create_engine(MYSQL_URL)
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def create_table_in_db(self):
        create_table_query = """
                    CREATE TABLE IF NOT EXISTS weatherdata (
                        date VARCHAR(100) PRIMARY KEY,
                        visibility float,
                        drybulbtempf int,
                        wetbulbtempf int,
                        dewpointtempf int,
                        relativehumidity int,
                        windspeed int,
                        winddirection int,
                        stationpressure float,
                        sealevelpressure float,
                        precip float
                    )
                """
        
        with self.engine.connect() as connection:
            connection.execute(create_table_query)

    def save_csv_into_db(self,file_path: str,table_name = TABLE_NAME):
        logging.info("Save dataset into Database")
        df = pd.read_csv(file_path)
        table_name = table_name
        df.to_sql(table_name,con = self.engine,if_exists='replace', index=False)

        count_query = """select count(*) from weatherdata"""
        with self.engine.connect() as connection:
            result = connection.execute(count_query)
            count = result.scalar()
        return count
    
    def extract_data_from_db(self):
        # Query the database and fetch the results into a DataFrame
        logging.info("Fetch data into Dataframe ")
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql_query(query, self.engine)
        
        return df
        # df.to_csv(filepath,index= False)

# if __name__ == "__main__":
#     cd = ClimateData()
#     cd.create_table_in_db()
#     # rows = cd.save_csv_into_db("C:\\Users\\Sheela Sai kumar\\Documents\\Untitled Folder\\Visibility-Climate\\data\\jfk_weather_cleaned.csv")
#     # print(rows)
#     # cd.extract_data_from_db("C:\\Users\\Sheela Sai kumar\\Documents\\Untitled Folder\\Visibility-Climate\\data\\data.csv")


    