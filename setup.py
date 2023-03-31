from setuptools import setup

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()
    setup(
        name='tensorslow',
        version='0.0.1',
        description=["A slower machine learning framework"],
        long_description=long_description,
        long_description_content_type="text/markdown",
    )

