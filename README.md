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
```
[onjyfans]$ slurp-test.sh
```

### Model Training

Train the model:
```
[onjyfans]$ slurp.sh
```

### Model Evaluation

Run evaluation:
```
[onjyfans]$ slurp-eval.sh
```

### Generate a prediction

Run prediction: 
```
[onjyfans]$ slurp-example.sh
```

This generates a CSV file containing:
- Input messages
- Expected outputs
- Generated outputs

### Further Evaluation

To evaluate the generated `evaluation_gams9b-02.csv` checkout Evaluation.ipynb

## Models Used

- **GaMS-1B**: Lightweight model for efficient traffic news generation
- **GaMS-9B-Instruct**: Larger instruction-tuned model for enhanced performance

## Files and Directories

```
ul-fri-nlp-course-project-2024-2025-onjyfans/
├── evaluation_gams9b.csv                             #copied from hpc
├── evaluation_gams9b-02.csv                          #copied from hpc
├── outputs9B/
├── LICENSE
├── README.md
├── requirements.txt                                  #pip install -r requirements.txt
├── report/
│   ├── code/
│   │   ├── DataProcessAndAnalisys.ipynb -            #local
│   │   ├── EvalFineTuning.py                         #used for hpc
│   │   ├── Evaluation.ipynb                          #local
│   │   ├── FineTuning.py                             #used for hpc
│   │   ├── FineTuningOnGoogleCollab.ipynb            #local
│   │   ├── FineTuningTest.py                         #used for hpc
│   │   ├── InitialAnalisys.ipynb                     #local
│   │   ├── PredictAnExample.py                       #used for hpc
│   │   └── data/
│   │       ├── Joined_rtf_files.csv                  #Csv file containing joined parsed rtf files
│   │       ├── PP_ALL.csv                            #Joined prometno porocilo (traffic report data)
│   │       └── PP_sample.csv                         #just a sample of traffic report
│   ├── results/
│   └── trainingdataset2.csv                          #Main training dataset
```

hpc 

```
onjyfans/
├── containers/                      # Singularity 
├── evaluation_gams9b.csv            # Evaluation results 
├── evaluation_gams9b-02.csv         # Evaluation results from a second run
├── logs/                            # Directory for storing logs
├── outputs/                         # Output directory from model training or inference
├── results/                         # Final evaluation results, metrics, etc.
├── slurp.sh                         # Main training script
├── slurp-example.sh                 # Example prediction script
├── slurp-test.sh                    # Script for testing environment before run
├── slurp-eval.sh                    # Evaluation script that creates-> evaluation_gams9b-02.csv  
├── training_logs/                   # Folder with multiple logs from different runs
├── evaluation_gams9b.csv            
├── evaluation_gams9b-02.csv         # Generate with slurp-eval.sh      

```

## Technical Requirements

- Python 3.9+
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
4. **Evaluation**: Run Evaluation.ipynb scripts to generate performance metrics and inspect generated evalutions during training
5. **Testing**: Use provided scripts on hpc to test random examples outputs

## Output

The project generates automated Slovenian traffic news suitable for RTV Slovenija, with comprehensive evaluation metrics comparing different model approaches and prompting strategies.