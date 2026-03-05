# Contributing to HUD

We welcome contributions to the HUD SDK! This guide covers how to get started.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/hud-python`
3. Install in development mode:
   ```bash
   cd hud-python
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests
```bash
pytest                     # Run all tests
```

### Code Quality
```bash
ruff check .               # Linting
ruff format .              # Formatting
pyright hud/               # Type checking
```

## Contribution Types

### 🐛 Bug Fixes
- Include a test that reproduces the issue
- Reference the issue number in your PR

### ✨ New Features
- Open an issue first to discuss the design
- Add tests and documentation
- Update relevant examples

### 🌍 New Environments
- Follow the structure in `environments/README.md`
- Must pass all 5 phases of `hud debug`
- Include comprehensive tests

### 📚 Documentation
- Fix typos, clarify explanations
- Add examples or diagrams
- Improve docstrings

## Pull Request Process

1. **Branch naming**: `feature/description` or `fix/issue-number`
2. **Commits**: Use clear, descriptive messages
3. **Tests**: All tests must pass
4. **Documentation**: Update if needed
5. **Review**: Address feedback promptly

## Environment Development

When contributing new environments, follow the comprehensive guide in [`environments/README.md`](environments/README.md). Key requirements:
- All environments must pass `hud debug <your-image>` (5 phases)
- Use `MCPServer` wrapper for Docker compatibility
- Implement standard `setup` and `evaluate` tools
- Include interaction tools for agent control

## Code Style

- Python 3.11+ features are allowed
- Type hints required for public APIs
- Follow existing patterns in the codebase
- Keep line length under 200 characters

## Need Help?

- Check existing issues and PRs
- Look at similar code in the repository
- Ask questions in your PR

> By contributing, you agree that your contributions will be licensed under the MIT License.