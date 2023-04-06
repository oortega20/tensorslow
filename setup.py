from setuptools import setup

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()
    setup(
        name="tensorslow",
        version="1.0.2",
        description="A slower machine learning framework",
        author="Oscar Ortega",
        author_email="oscar.g.ortega.5@gmail.com",
        url="https://github.com/oortega20/tensorslow/",
        long_description=long_description,
        long_description_content_type="text/markdown",
    )

