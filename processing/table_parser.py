"""
Table parsing and analysis.

Converts extracted tables to structured DataFrames and generates summaries.
"""

from typing import Dict, List, Optional, Any
import io

import pandas as pd

from utils.logger import setup_logger
from models.document import TableAnalysis

logger = setup_logger(__name__)


class TableParser:
    """Parses and analyzes table structures."""

    @staticmethod
    def table_to_dataframe(
        table_data: List[List[Optional[str]]],
    ) -> Optional[pd.DataFrame]:
        """
        Convert table data to pandas DataFrame.

        Args:
            table_data: 2D list of table cells.

        Returns:
            pandas DataFrame or None if conversion fails.
        """
        try:
            if not table_data or len(table_data) == 0:
                logger.warning("Empty table data provided")
                return None

            # Use first row as header if available
            df = pd.DataFrame(table_data[1:], columns=table_data[0])

            logger.debug(f"Created DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")
            return df

        except Exception as e:
            logger.warning(f"Failed to convert table to DataFrame: {str(e)}")
            return None

    @staticmethod
    def dataframe_to_csv(df: pd.DataFrame) -> str:
        """
        Convert DataFrame to CSV string.

        Args:
            df: pandas DataFrame.

        Returns:
            CSV-formatted string.
        """
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
        except Exception as e:
            logger.warning(f"Failed to convert DataFrame to CSV: {str(e)}")
            return ""

    @staticmethod
    def generate_table_summary(
        df: pd.DataFrame,
        max_rows: int = 5,
    ) -> str:
        """
        Generate a natural language summary of a table.

        Args:
            df: pandas DataFrame.
            max_rows: Maximum number of rows to include in summary.

        Returns:
            Natural language summary text.
        """
        try:
            if df.empty:
                return "Empty table"

            summary_parts = [
                f"Table with {len(df)} rows and {len(df.columns)} columns."
            ]

            # Column information
            columns_str = ", ".join(df.columns.astype(str))
            summary_parts.append(f"Columns: {columns_str}")

            # Data type information
            dtypes_info = []
            for col in df.columns:
                try:
                    # Try to infer numeric type
                    numeric_df = pd.to_numeric(df[col], errors="coerce")
                    if numeric_df.notna().sum() > len(df) * 0.5:  # Mostly numeric
                        min_val = numeric_df.min()
                        max_val = numeric_df.max()
                        dtypes_info.append(
                            f"{col}: {min_val:.2f} to {max_val:.2f}"
                        )
                    else:
                        dtypes_info.append(f"{col}: text")
                except:
                    dtypes_info.append(f"{col}: text")

            if dtypes_info:
                summary_parts.append("Value ranges: " + ", ".join(dtypes_info))

            # Sample data
            sample_rows = df.head(max_rows)
            summary_parts.append("\nSample rows:")
            summary_parts.append(sample_rows.to_string())

            return "\n".join(summary_parts)

        except Exception as e:
            logger.warning(f"Failed to generate table summary: {str(e)}")
            return "Failed to summarize table"

    @staticmethod
    def extract_table_metadata(
        table_data: List[List[Optional[str]]],
    ) -> Optional[TableAnalysis]:
        """
        Extract metadata and analysis from table data.

        Args:
            table_data: 2D list of table cells.

        Returns:
            TableAnalysis object or None if extraction fails.
        """
        try:
            if not table_data:
                return None

            rows = len(table_data)
            cols = len(table_data[0]) if table_data else 0
            headers = table_data[0] if table_data else []

            # Convert to DataFrame for analysis
            df = TableParser.table_to_dataframe(table_data)
            summary = ""
            key_metrics = []

            if df is not None:
                summary = TableParser.generate_table_summary(df)

                # Extract numeric columns for metrics
                for col in df.columns:
                    try:
                        numeric_col = pd.to_numeric(df[col], errors="coerce")
                        if numeric_col.notna().sum() > 0:
                            key_metrics.append(
                                f"{col}: avg={numeric_col.mean():.2f}, "
                                f"min={numeric_col.min():.2f}, "
                                f"max={numeric_col.max():.2f}"
                            )
                    except:
                        pass

            return TableAnalysis(
                table_summary=summary,
                row_count=rows,
                column_count=cols,
                headers=headers,
                key_metrics=key_metrics if key_metrics else None,
            )

        except Exception as e:
            logger.warning(f"Failed to extract table metadata: {str(e)}")
            return None
