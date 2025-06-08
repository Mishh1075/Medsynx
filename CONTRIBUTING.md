# Contributing to MedSynX

We love your input! We want to make contributing to MedSynX as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/yourusername/medsynx/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/medsynx/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

1. Set up your development environment:
```bash
# Clone repository
git clone https://github.com/yourusername/medsynx.git
cd medsynx

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes:
- Write clean, documented code
- Follow PEP 8 style guide
- Add tests for new functionality

4. Run tests:
```bash
pytest tests/
```

5. Run linting:
```bash
black app/ frontend/
flake8 app/ frontend/
```

6. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

7. Push to your fork:
```bash
git push origin feature/your-feature-name
```

8. Create a Pull Request

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting
- Write docstrings for all public functions
- Comment complex logic
- Use type hints

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Maintain test coverage above 80%
- Use pytest fixtures for common test setups

## Documentation

- Update README.md if needed
- Document new features
- Update API documentation
- Include docstrings for new functions/classes

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 