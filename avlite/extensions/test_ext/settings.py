class ExtensionSettings:
    exclude = ["exclude", "filepath"] # attributes to exclude from saving/loading
    filepath: str="configs/ext_test.yaml"

    test: str = "test_value"
