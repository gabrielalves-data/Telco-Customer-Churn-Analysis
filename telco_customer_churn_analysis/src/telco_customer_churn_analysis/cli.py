"""Console script for telco_customer_churn_analysis."""

import typer
from rich.console import Console

from telco_customer_churn_analysis import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for telco_customer_churn_analysis."""
    console.print("Replace this message by putting your code into "
               "telco_customer_churn_analysis.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
