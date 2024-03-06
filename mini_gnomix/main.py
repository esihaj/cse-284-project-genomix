import typer

app = typer.Typer()

@app.command()
def greet(name: str = typer.Argument(None, help="Your name")):
    """Greets the user with a name, if provided."""
    greeting = f"Hello, {name}" if name else "Hello, World!"
    typer.echo(greeting)

if __name__ == "__main__":
    app()
