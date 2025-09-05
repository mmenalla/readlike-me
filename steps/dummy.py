from zenml import step

@step
def dummy_start() -> str:
    print("Starting pipeline...")
    return "start"

@step
def dummy_end() -> None:
    print(f"Pipeline ended.")