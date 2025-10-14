# Contributing Guidelines

Thank you for your interest in contributing to this Financial Time Series Forecasting project!

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/ML-Intern.git
   cd ML-Intern
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Follow the existing code style
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write unit tests for new features

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py

# Check code coverage
python -m pytest tests/ --cov=src
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
# or
git commit -m "fix: bug fix description"
```

**Commit Message Format**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `style:` Code style changes (formatting, etc.)

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Code

- **PEP 8**: Follow Python style guide
- **Type Hints**: Use type annotations
  ```python
  def train_model(model: nn.Module, data: torch.Tensor) -> Dict[str, float]:
      ...
  ```
- **Docstrings**: Use Google-style docstrings
  ```python
  def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
      """
      Calculate Relative Strength Index.
      
      Args:
          prices: Array of closing prices
          window: Lookback period (default: 14)
          
      Returns:
          Array of RSI values
      """
  ```

### File Organization

- Keep files under 500 lines
- One class per file when possible
- Group related functions

### Configuration Files

- Use YAML for configurations
- Include comments explaining parameters
- Provide default values

## Testing Guidelines

### Writing Tests

- Test files should mirror source structure (`test_models.py` for `models/`)
- Use descriptive test names: `test_lstm_forward_pass_correct_shape()`
- Include edge cases and error conditions

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths first
- Test error handling

## Documentation

### Code Documentation

- All functions need docstrings
- Complex algorithms need inline comments
- Update README.md if adding features

### Report Documentation

When adding features that affect results:
- Update `reports/PROJECT_ANALYSIS_REPORT.md`
- Add figures to `reports/figures/`
- Update relevant tables in `reports/tables/`

## Adding New Models

1. **Create model file**: `src/models/your_model.py`
2. **Implement base interface**:
   ```python
   class YourModel(nn.Module):
       def __init__(self, input_dim: int, ...):
           ...
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           ...
   ```
3. **Add configuration**: `configs/your_model.yaml`
4. **Write tests**: `tests/test_your_model.py`
5. **Update documentation**: Add to README.md and reports

## Adding New Features

### Technical Indicators

1. **Add to** `src/data/indicators.py`
2. **Update** `add_technical_indicators()` function
3. **Document** in `reports/tables/TECHNICAL_INDICATORS_TABLE.md`
4. **Test** in `tests/test_preprocess.py`

### Evaluation Metrics

1. **Add to** `src/evaluation/metrics.py`
2. **Write unit tests**
3. **Document** purpose and formula

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No unnecessary files included
- [ ] Descriptive PR title and description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Screenshots (if applicable)
Add screenshots for UI changes
```

## Code Review Process

1. **Automated Checks**: CI/CD runs tests
2. **Code Review**: Maintainer reviews code
3. **Feedback**: Address review comments
4. **Approval**: Once approved, PR is merged

## Questions or Issues?

- **Author**: Mohansree Vijayakumar
- **Email**: mohansreesk14@gmail.com
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Open an issue with [Feature Request] tag
- **Questions**: Use GitHub Discussions or issues

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers and help them learn
- Keep discussions professional

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing! 🎉
