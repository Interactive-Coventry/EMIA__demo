import typer

app = typer.Typer()

from os.path import join as pathjoin


@app.command()
def test():
    """
    Test functionalities for provide_insights using default images.
    """

    import provide_insights

    results = provide_insights.test_analyze()
    print('Test success.')


@app.command()
def analyze(filename: str = typer.Argument(help="Target image filename."),
            history_length: int = typer.Argument(5, help="The lookback window.")):
    """
    Get insights for a target image using data for the past N time steps as history.
    """

    import provide_insights

    results = provide_insights.analyze(filename, delete_previous_results=False,
                                       history_length=history_length)

    provide_insights.print_results(results)
    print(f'\nExecution was successful.')
    return results


if __name__ == "__main__":
    app()
