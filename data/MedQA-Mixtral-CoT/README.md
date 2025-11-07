---
license: apache-2.0
language:
- en
tags:
- medical
- biology
size_categories:
- 10K<n<100K
task_categories:
- multiple-choice
- question-answering
---

# Dataset Card for medqa-cot

<!-- Provide a quick summary of the dataset. -->

Synthetically enhanced responses to the medqa dataset using mixtral.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

To increase the quality of answers from the training splits of the [MedQA](https://github.com/jind11/MedQA) dataset, we leverage Mixtral-8x7B to generate Chain of Thought(CoT) answers. We create a custom prompt for the dataset, along with a
hand-crafted list of few-shot examples. For a multichoice answer, we ask the model to rephrase and explain the question, then explain each option with respect to the question, then summarise this explanation to arrive at the final solution. During this synthetic data generation process, the model is also given the solution
and the reference answer. For the cases where the model fails to generate correct responses and just reiterates the input question, we regenerate the solutions until a correct response is generated. More details are available in the paper.

- **Curated by:** [Ashwin Kumar Gururajan](https://huggingface.co/G-AshwinKumar)
- **Language(s) (NLP):** English
- **License:** Apache 2.0

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Paper:** [Aloe: A Family of Fine-tuned Open Healthcare LLMs](https://arxiv.org/abs/2405.01886)

## Dataset Creation

### Curation Rationale

This dataset was created to provide a high quality easy to use instruction tuning dataset based on medqa.

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@misc{gururajan2024aloe,
      title={Aloe: A Family of Fine-tuned Open Healthcare LLMs}, 
      author={Ashwin Kumar Gururajan and Enrique Lopez-Cuena and Jordi Bayarri-Planas and Adrian Tormos and Daniel Hinjos and Pablo Bernabeu-Perez and Anna Arias-Duart and Pablo Agustin Martin-Torres and Lucia Urcelay-Ganzabal and Marta Gonzalez-Mallo and Sergio Alvarez-Napagao and Eduard Ayguadé-Parra and Ulises Cortés Dario Garcia-Gasulla},
      year={2024},
      eprint={2405.01886},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```

## Dataset Card Authors

[Ashwin Kumar Gururajan](https://huggingface.co/G-AshwinKumar)

## Dataset Card Contact

[hpai@bsc.es](mailto:hpai@bsc.es)