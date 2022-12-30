# Submission to the QuALITY Leaderboard

Last updated: 12/30/2022


## What to submit

1. [Required] Submitter's name and organization.

2. [Required] Contact email, and whether you want to make it public.

3. [Required] Prediction file. 
- Format: The file should contain 2128 lines. Each line corresponds to one question. Each line is composed of the question id (e.g., `52845_75VB1ISR_1`), a comma, and the prediction (e.g., `3`). The prediction is 1-indexed, so the first option corresponds to `1` and the fourth option corresponds to `4`. 
  - You can also abstain from making a prediction, by writing `-1` as the option. 
- Example line: `52845_75VB1ISR_1,3` or `20001_G5VZ19S0_1,-1`
- We will report both the accuracy and the SAT-style score which penalizes incorrect predictions (see FAQ section). 

4. [Required] Short model description. This information will appear on the front page of the leaderboard.
- Example: "RoBERTa-large with DPR-based extraction, with intermediate training on RACE"

5. [Required] Long model description (plain text or URL to pdf; at least a paragraph). After reading the model description, an experienced NLP researcher/engineer should have a rough idea of how to implement the model. If you are providing the URL to your paper below AND the model in the paper can be directly used for the QuALITY dataset, then you do not need to write a comprehensive description.

6. [Optional but encouraged] What external data or resources (e.g., off-the-shelf pretrained language model, retrieval model, knowledge graph, other datasets, etc.) did you use to train the system? 
- Example 1: "We use the pretrained DeBERTa-v3-base model on Hugging Face. We do intermediate training on RACE (Lai et al., 2017), and then fine-tune on QuALITY's training set. During fine-tuning on QuALITY, we use the off-the-shelf DPR retriever (Karpukhin et al., 2020)."
- Example 2: "We train a custom version of T5 on BookCorpus and Wikipedia. We fine-tune on QuALITY; during fine-tuning, we use ConceptNet 5.5 (Speer et al., 2016)."

7. [Optional but encouraged] URL to your publicly accessible codebase (e.g., Github repo).

8. [Optional] URL to the model checkpoint. 

9. [Optional] URL to the associated paper. If the paper does not include model information specific to QuALITY, then please remember fill out the long model description carefully. 

10. [Optional] Comments.


## How to submit

Please fill out [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSdFBTnD-RoND30qrchQJTps2AGCrpx4h1T9IQNAgyxadFzZ9Q/viewform?usp=sf_link).

If it is infeasible for you to submit through the Google form, then please send an email containing the above information to quality-benchmark-admins@googlegroups.com.


## FAQ

1. What evaluation metrics will be used?
- We will measure the model performance by accuracy. Two accuracy numbers will be computed: the accuracy on the entire test set and the accuracy on the hard subset of the test set. 
- We will also report the SAT-style score which penalizes incorrect predictions. The score equals (number of correct answers - (1/3) * number of incorrect answers + 0 * number of abstained answers) / (number of examples). A set of random predictions would receive a score close to 0. Therefore, if your model is very uncertain on an example, it would be best to abstain according to this metric.

2. Can I make an anonymous submission to the leaderboard?
- Yes, but please try not to do so unless necessary. 

<!-- Anonymous submissions will be displayed on the leaderboard only if (1) the submitter submits a link to a reasonably detailed and possibly anonymized paper which is publicly accessible (e.g., on OpenReview), and (2) the submitter indicates that the submitter is an author of the paper.  -->

3. When will my results appear on the leaderboard?
- Assuming your submission is eligible for the leaderboard, we will try to update the leaderboard within a week, and often much more promptly. 
