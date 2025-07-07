# Documentation Guide for Splinator

This guide explains how to create and host documentation for your Python package using free tools.

## Documentation Setup

We've set up Sphinx documentation with the following features:
- **Automatic API documentation** from your Python docstrings
- **Read the Docs theme** for a professional look
- **Type hints support** in documentation
- **Examples and tutorials** sections

## Building Documentation Locally

1. **Install documentation dependencies:**
   ```bash
   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
   ```

2. **Build the documentation:**
   ```bash
   cd docs
   make html
   ```

3. **View the documentation:**
   Open `docs/_build/html/index.html` in your browser

## Free Hosting Options

### Option 1: Read the Docs (Recommended)

Read the Docs provides free hosting for open source projects with automatic builds.

**Setup steps:**

1. Go to [readthedocs.org](https://readthedocs.org/) and sign up
2. Connect your GitHub/GitLab account
3. Import your repository
4. Read the Docs will automatically detect `.readthedocs.yaml` and build your docs
5. Your documentation will be available at `https://splinator.readthedocs.io/`

**Features:**
- Automatic builds on every push
- Version management (stable, latest, tags)
- Search functionality
- PDF and ePub downloads
- Custom domains (splinator.readthedocs.io)

### Option 2: GitHub Pages

Host documentation directly from your GitHub repository.

**Setup steps:**

1. Build documentation locally:
   ```bash
   cd docs
   make html
   ```

2. Create a `gh-pages` branch:
   ```bash
   git checkout --orphan gh-pages
   git rm -rf .
   cp -r docs/_build/html/* .
   touch .nojekyll  # Important for Sphinx
   git add .
   git commit -m "Initial documentation"
   git push origin gh-pages
   ```

3. Enable GitHub Pages in repository settings:
   - Go to Settings → Pages
   - Select source: "Deploy from a branch"
   - Select branch: `gh-pages` and folder: `/ (root)`

4. Documentation will be available at `https://yourusername.github.io/splinator/`

**Automated GitHub Pages with Actions:**

Create `.github/workflows/docs.yml`:

```yaml
name: Build and Deploy Documentation

on:
  push:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r docs/requirements.txt
        
    - name: Build documentation
      run: |
        cd docs
        make html
        
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

### Option 3: GitLab Pages

Similar to GitHub Pages but for GitLab repositories.

Create `.gitlab-ci.yml`:

```yaml
pages:
  stage: deploy
  image: python:3.11
  script:
    - pip install -r docs/requirements.txt
    - cd docs
    - make html
    - mv _build/html ../public
  artifacts:
    paths:
      - public
  only:
    - main
```

## Documentation Best Practices

1. **Write good docstrings:**
   ```python
   def calibrate(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
       """Calibrate probability predictions.
       
       Parameters
       ----------
       probabilities : np.ndarray
           Uncalibrated probability predictions
       labels : np.ndarray
           True binary labels
           
       Returns
       -------
       np.ndarray
           Calibrated probabilities
       """
   ```

2. **Keep documentation updated:**
   - Update docstrings when changing functions
   - Add examples for new features
   - Update version numbers

3. **Use type hints:**
   They automatically appear in documentation with `sphinx-autodoc-typehints`

4. **Add examples:**
   - In docstrings: Use the `Examples` section
   - In documentation: Create example notebooks

## Current Documentation Structure

```
docs/
├── conf.py           # Sphinx configuration
├── index.rst         # Main page
├── installation.rst  # Installation guide
├── quickstart.rst    # Quick start guide
├── api.rst          # API reference (auto-generated)
├── examples.rst     # Examples and tutorials
└── requirements.txt # Documentation dependencies
```

## Troubleshooting

**Issue: Import errors when building docs**
- Solution: We use `autodoc_mock_imports` in `conf.py` to mock external dependencies

**Issue: Documentation not updating on Read the Docs**
- Check build logs on Read the Docs dashboard
- Ensure `.readthedocs.yaml` is in the repository root

**Issue: Styles not loading on GitHub Pages**
- Add `.nojekyll` file to the documentation root
- Ensure `_static` folder is included

## Next Steps

1. Choose your hosting platform (Read the Docs recommended)
2. Set up the hosting following the steps above
3. Add a documentation badge to your README:
   ```markdown
   [![Documentation Status](https://readthedocs.org/projects/splinator/badge/?version=latest)](https://splinator.readthedocs.io/en/latest/?badge=latest)
   ```

Your documentation is now ready to be built and hosted for free! 