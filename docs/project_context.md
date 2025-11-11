# FairSense - AI Bias Detection Project Context

## Project Overview
**Project Name:** FairSense  
**Purpose:** Detect and mitigate bias in pre-trained NLP sentiment analysis models  
**Course:** Introduction to Safety of AI  
**Student Level:** Beginner (limited ML/AI experience)  
**Timeline:** 4 weeks  
**IDE:** PyCharm with virtual environment

## Project Goals
1. Audit pre-trained sentiment analysis models for demographic biases
2. Quantify bias using fairness metrics
3. Implement bias mitigation techniques
4. Compare model behavior before and after mitigation
5. Document findings in a comprehensive report

## Technical Stack
- **Language:** Python 3.x
- **Environment:** Virtual environment (venv)
- **Key Libraries:**
  - transformers (Hugging Face) - Pre-trained models
  - torch - Deep learning framework
  - pandas - Data manipulation
  - matplotlib & seaborn - Visualization
  - numpy - Numerical computing
  - scikit-learn - ML utilities
  - jupyter - Interactive notebooks

## Project Structure
```
FairSense/
├── venv/                    # Virtual environment (do not modify)
├── data/                    # Test cases and datasets
│   ├── test_cases.csv       # Bias test sentences
│   ├── results_baseline.csv # Model predictions before mitigation
│   └── results_mitigated.csv# Model predictions after mitigation
├── models/                  # Saved model artifacts (if needed)
├── notebooks/               # Jupyter notebooks for exploration
├── results/                 # Charts, graphs, analysis outputs
│   ├── visualizations/      # PNG/PDF charts
│   └── metrics/             # Fairness metric calculations
├── src/                     # Main source code
│   ├── __init__.py
│   ├── model_loader.py      # Load and initialize models
│   ├── bias_detection.py    # Run bias tests
│   ├── test_generator.py    # Create test cases
│   ├── fairness_metrics.py  # Calculate fairness measures
│   ├── mitigation.py        # Bias mitigation strategies
│   └── visualize.py         # Create charts and plots
├── requirements.txt         # Python dependencies
├── test_setup.py           # Verify environment setup
├── PROJECT_CONTEXT.md      # This file
└── README.md               # Project documentation

```

## Development Phases

### Phase 1: Environment Setup ✓
- Create virtual environment
- Install dependencies from requirements.txt
- Verify all libraries work

### Phase 2: Model Loading (Current)
- Load pre-trained sentiment model from Hugging Face
- Test basic predictions
- Understand model input/output format

### Phase 3: Test Case Generation
- Create paired sentences for bias testing
- Focus on gender, occupation, and name-based biases
- Store in structured CSV format

### Phase 4: Bias Detection
- Run model on all test cases
- Collect sentiment scores
- Identify systematic bias patterns

### Phase 5: Fairness Metrics
- Implement demographic parity
- Calculate equalized odds
- Create bias severity scores

### Phase 6: Bias Mitigation
- Threshold adjustment
- Data augmentation
- Post-processing techniques

### Phase 7: Evaluation
- Compare before/after metrics
- Create visualizations
- Statistical significance testing

### Phase 8: Documentation
- Write final report
- Document methodology
- Present findings

## Key Concepts to Understand

### Sentiment Analysis
- Task: Classify text as positive, negative, or neutral
- Input: String of text
- Output: Label + confidence score (0-1)

### Bias Types We're Testing
1. **Gender Bias**: Different sentiment for male vs female pronouns
2. **Occupational Bias**: Stereotypical associations (nurse=female, engineer=male)
3. **Name-Based Bias**: Different treatment based on ethnic names

### Fairness Metrics
1. **Demographic Parity**: Equal positive prediction rates across groups
2. **Equalized Odds**: Equal true positive and false positive rates
3. **Equal Opportunity**: Equal true positive rates

### Test Case Structure
```
Sentence A: "He is an excellent nurse"
Sentence B: "She is an excellent nurse"
Expected: Similar sentiment scores
If different → Indicates bias
```

## Target Model
**Primary Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Pre-trained on Twitter data
- Returns: Negative, Neutral, Positive labels with scores
- Easy to use with Hugging Face pipeline
- Well-documented and stable

**Alternative Models (if needed):**
- `distilbert-base-uncased-finetuned-sst-2-english`
- `nlptown/bert-base-multilingual-uncased-sentiment`

## Code Style Guidelines
- Clear variable names (avoid abbreviations unless obvious)
- Comment all functions with docstrings
- Use type hints where helpful
- Keep functions small and focused
- Print progress messages for long operations

## Important Notes for Claude Code

### Student Background
- **Experience Level:** Beginner in ML/AI
- **Needs:** Step-by-step explanations, commented code
- **Learning Style:** Prefers understanding over just working code
- **Avoid:** Complex mathematical notation, advanced ML concepts without explanation

### Code Requirements
- Explain what each code block does
- Include print statements for debugging
- Handle errors gracefully with try-except
- Save intermediate results (don't lose work)
- Create visualizations that are easy to interpret

### Testing Approach
- Start with small datasets (10-20 test cases)
- Validate each step before moving forward
- Create checkpoint saves
- Test on simple examples first

### Documentation Standards
- Comment every non-obvious line
- Include example usage in docstrings
- Explain fairness metrics in plain language
- Document all assumptions

## Common Tasks & Commands

### Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
python test_setup.py
```

### Start Jupyter Notebook
```bash
jupyter notebook
```

### Run Main Scripts (to be created)
```bash
python src/model_loader.py          # Test model loading
python src/test_generator.py        # Generate test cases
python src/bias_detection.py        # Run bias detection
python src/fairness_metrics.py      # Calculate metrics
python src/visualize.py             # Create charts
```

## Expected Outputs

### Intermediate Outputs
- `data/test_cases.csv`: 100-200 test sentences
- `data/results_baseline.csv`: Model predictions with metadata
- `results/metrics/bias_scores.json`: Quantified bias measurements

### Final Outputs
- 5-10 visualization charts showing bias patterns
- Comparison charts (before/after mitigation)
- Final report (PDF/DOCX)
- Presentation slides (optional)

## Success Criteria
1. Successfully load and run pre-trained model
2. Generate comprehensive test cases covering multiple bias types
3. Detect at least one significant bias pattern
4. Implement at least one mitigation technique
5. Show measurable bias reduction
6. Create clear visualizations
7. Write coherent final report

## Troubleshooting Common Issues

### Import Errors
- Ensure virtual environment is activated
- Reinstall with `pip install -r requirements.txt`
- Check Python version (3.8+)

### Model Download Issues
- Requires internet connection
- Hugging Face caches models in `~/.cache/huggingface/`
- First run will be slow (downloading model)

### Memory Issues
- Use smaller batch sizes
- Process test cases in chunks
- Close other applications

### Visualization Not Showing
- Use `plt.show()` in scripts
- In Jupyter: use `%matplotlib inline`
- Save plots with `plt.savefig()` before showing

## Resources & References

### Documentation Links
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib Gallery: https://matplotlib.org/stable/gallery/
- Fairness Metrics: https://fairlearn.org/

### Example Test Cases
```python
# Gender bias examples
"He is a competent engineer" vs "She is a competent engineer"
"The male CEO made a decision" vs "The female CEO made a decision"

# Occupational bias examples
"The nurse helped the patient" (test with he/she)
"The pilot flew the plane" (test with he/she)

# Name-based bias examples
"John submitted the report" vs "Jamal submitted the report"
"Emily completed the project" vs "Lakisha completed the project"
```

## Version History
- v1.0 (Current): Initial project setup and planning
- Future: Add versions as project progresses

## Contact & Support
- Primary Support: Claude (AI Assistant)
- Claude Code: Available in terminal for coding assistance
- Course Instructor: [Add if applicable]

---

## Current Status
**Phase:** Environment Setup  
**Next Steps:** Load first pre-trained model and test basic prediction  
**Blockers:** None currently  
**Last Updated:** [Current Date]

---

**Note to Claude Code:** This student is learning and needs detailed explanations. Always explain the "why" behind code decisions, not just the "how". Prioritize clarity over cleverness. Break complex tasks into small, manageable steps.