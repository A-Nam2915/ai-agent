from pathlib import Path
from io import StringIO
import sqlite3
import pandas as pd


CSV_PATH = Path("nba_player_data.csv") 
DATABASE_PATH = Path("nba_analytics.db")

def create_mock_csv_if_missing(csv_path: Path) -> None:
    """
    Create a mock NBA CSV file if the real file does not exist.
    """

    if csv_path.exists():
        print(f"Using existing file: {csv_path}")
        return

    mock_csv = """Player,Season,Team,Pos,Age,GP,MP,PTS,AST,TRB,STL,BLK,TOV,PER,WS,BPM,VORP,Salary
Marcus Reed,2024-25,DAL,PG,25,78,34.2,22.8,8.1,4.5,1.6,0.3,2.7,22.4,9.8,5.7,4.3,18500000
Darius Coleman,2024-25,CHI,SF,30,69,32.5,18.6,3.4,6.8,1.2,0.7,2.1,17.8,6.2,1.9,2.1,28750000
Ethan Brooks,2024-25,ORL,C,23,74,27.8,13.2,2.0,9.7,0.8,1.8,1.6,20.1,8.4,3.2,2.9,6800000
Jamal Foster,2024-25,LAL,SG,34,12,18.6,8.7,2.1,2.4,0.5,0.1,1.3,10.2,0.4,-2.8,-0.1,14200000
Noah Bennett,2024-25,BOS,PF,27,81,35.1,20.4,4.7,8.2,1.1,1.0,2.0,21.7,11.3,4.8,4.7,24000000
"""

    mock_dataframe = pd.read_csv(StringIO(mock_csv))

    mock_dataframe.to_csv(
        csv_path,
        index=False,
    )

    print(f"Mock CSV created: {csv_path}") 

def load_player_data(csv_path: Path) -> pd.DataFrame:
    """
    Read the NBA player CSV file into a pandas DataFrame.
    """

    dataframe = pd.read_csv(csv_path)

    print(f"Loaded {len(dataframe)} player records.")
    print(f"Columns: {list(dataframe.columns)}")

    return dataframe

def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Check that the dataset contains the columns needed for analysis.
    """

    required_columns = {
        "Player",
        "GP",
        "PTS",
        "PER",
        "WS",
        "BPM",
        "VORP",
        "Salary",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print("All required columns are present.") 

def clean_player_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the NBA player dataset before loading it into SQLite.
    """

    cleaned_dataframe = dataframe.copy()

    cleaned_dataframe = cleaned_dataframe.drop_duplicates()

    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=[
            "Player",
            "GP",
            "PER",
            "WS",
            "BPM",
            "VORP",
            "Salary",
        ]
    )

    cleaned_dataframe["Salary"] = cleaned_dataframe["Salary"].astype(float)

    cleaned_dataframe["GP"] = cleaned_dataframe["GP"].astype(int)

    print(f"Remaining rows after cleaning: {len(cleaned_dataframe)}")

    return cleaned_dataframe

def connect_to_database(database_path: Path) -> sqlite3.Connection:
    """
    Create a connection to the SQLite database.
    """

    connection = sqlite3.connect(database_path)

    print(f"Connected to database: {database_path}")

    return connection

if __name__ == "__main__":
    create_mock_csv_if_missing(CSV_PATH)

    player_dataframe = load_player_data(CSV_PATH)

    validate_required_columns(player_dataframe)

    cleaned_dataframe = clean_player_data(player_dataframe)

    database_connection = connect_to_database(DATABASE_PATH)