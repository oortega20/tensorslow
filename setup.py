from setuptools import setup

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()
    project_urls = {
        'GitHub': "https://github.com/oortega20/tensorslow"
    }

    setup(
        name='tensorslow',
        version='0.0.1',
        description=["A slower machine learning framework"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        project_urls=project_urls
    )

