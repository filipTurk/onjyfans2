# Natural Language Processing Course: Project 6

## Automatic Generation of Slovenian Traffic News for RTV Slovenija

This project focuses on generating Slovenian traffic news using large language models (LLMs). The pipeline includes data analysis, dataset creation, fine-tuning, and evaluation of language models such as GaMS-1B and GaMS-9B-Instruct.

## Project Structure

### Data Analysis and Processing

#### Initial Analysis
- **Location**: `report/code/InitialAnalisys.ipynb`
- **Purpose**: Initial corpus analysis and exploration of the traffic news dataset

#### Data Processing and Analysis
- **Location**: `report/code/DataProcessAndAnalisys.ipynb`
- **Purpose**: Further analysis and training dataset creation

### Model Training and Evaluation

#### GaMS-1B Model Training
- **Location**: `report/code/FineTuningOnGoogleCollab.ipynb`
- **Platform**: Google Colab
- **Methods**: 
  - Fine-tuning
  - One-shot prompting
  - Few-shot prompting

#### GaMS-9B-Instruct Model Training
- **Platform**: ARNES HPC Cluster
- **Location**: `onj_fri/onjyfans`

## Usage Instructions

### Environment Setup (ARNES HPC)

Test the environment:
```bash
./slurp-test.sh
```

### Model Training

Train the model:
```bash
./slurp.sh
```

### Model Evaluation

Run evaluation:
```bash
./slurp-Eval.sh
```

This generates a CSV file containing:
- Input messages
- Expected outputs
- Generated outputs

### Further Evaluation

To evaluate the generated `evaluation_dataset.csv`:
```bash
# TODO: Add command for further evaluation
```

### Testing Random Examples

To test with random examples:
```bash
# TODO: Add command for random testing
```

## Models Used

- **GaMS-1B**: Lightweight model for efficient traffic news generation
- **GaMS-9B-Instruct**: Larger instruction-tuned model for enhanced performance

## Files and Directories

```
project/
├── report/
│   └── code/
│       ├── InitialAnalisys.ipynb
│       ├── DataProcessAndAnalisys.ipynb
│       └── FineTuningOnGoogleCollab.ipynb
├── onj_fri/onjyfans/          # ARNES HPC training directory
├── slurp-test.sh              # Environment testing script
├── slurp.sh                   # Training script
├── slurp-Eval.sh              # Evaluation script
└── evaluation_dataset.csv     # Generated evaluation results
```

## Technical Requirements

- Python 3.8+
- Jupyter Notebook environment
- Google Colab access (for GaMS-1B training)
- ARNES HPC cluster access (for GaMS-9B-Instruct training)
- Required Python packages (see individual notebooks for dependencies)

## Workflow

1. **Initial Analysis**: Run `InitialAnalisys.ipynb` to explore the corpus
2. **Data Processing**: Execute `DataProcessAndAnalisys.ipynb` to create training datasets
3. **Model Training**: 
   - For GaMS-1B: Use Google Colab notebook
   - For GaMS-9B-Instruct: Use ARNES HPC cluster scripts
4. **Evaluation**: Run evaluation scripts to generate performance metrics
5. **Testing**: Use provided scripts to test model outputs

## Output

The project generates automated Slovenian traffic news suitable for RTV Slovenija, with comprehensive evaluation metrics comparing different model approaches and prompting strategies.