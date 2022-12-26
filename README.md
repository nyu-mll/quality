# QuALITY: Question Answering with Long Input Texts, Yes!

**Authors**: Richard Yuanzhe Pang,* Alicia Parrish,* Nitish Joshi,* Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman
(* = equal contribution)

## UPDATE

- [June 2022] Releasing QuALITY-v1.0.1 (minor update from v1.0: see [release notes](#release-notes)).
- [June 2022] Releasing QuALITY-v1.0 (minor update from v0.9: see [release notes](#release-notes)).
- [March 2022] [The leaderboard](https://nyu-mll.github.io/quality/) is up! 


## Data link

Download [QuALITY v1.0.1](https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip).

### Release notes

- [QuALITY-v1.0.1] Minor update. The `QuALITY.v1.0.1.[train/dev/test]` files are the same as `QuALITY.v0.9.[train/dev/test]` and `QuALITY.v1.0.[train/dev/test]`. The HTML-stripped files `QuALITY.v1.0.1.htmlstripped.[train/dev/test]` are slightly different from `QuALITY.v1.0.htmlstripped.[train/dev/test]`. We find that different articles may use different HTML tags to do paragraph or line breaks. So whenever `</p> <p>` or `</p><p>` or `<br/> <br/>` or `<br/><br/>` appears, we replace such tags with line breaks as well in the HTML-stripped files. 
- [QuALITY-v1.0] Minor update. The `QuALITY.v1.0.[train/dev/test]` files are the same as `QuALITY.v0.9.[train/dev/test]`. We fixed a small HTML-stripping issue ([details](#log)), so the `QuALITY.v1.0.htmlstripped.[train/dev/test]` files are slightly different from `QuALITY.v0.9.htmlstripped.[train/dev/test]`. 

## Paper

You can read the paper [here](https://arxiv.org/pdf/2112.08608.pdf).

## Leaderboard

https://nyu-mll.github.io/quality/

## Data README

Here are the explanations to the fields in the jsonl file. Each json line corresponds to the set of validated questions, corresponding to one article, written by one writer. 
- `article_id`: String. A five-digit number uniquely identifying the article. In each split, there are exactly two lines containing the same `article_id`, because two writers wrote questions for the same article.
- `set_unique_id`: String. The unique ID corresponding to the set of questions, which corresponds to the line of json. Each set of questions is written by the same writer.
- `batch_num`: String. The batch number. Our data collection is split in two groups, and there are three batches in each group. `[i][j]` means the j-th batch in the i-th group. For example, `23` corresponds to the third batch in the second group.
- `writer_id`: String. The anonymized ID of the writer who wrote this set of questions. 
- `source`: String. The source of the article. 
- `title`: String. The title of the article.
- `author`: String. The author of the article.
- `topic`: String. The topic of the article.
- `url`: String. The URL of the original unprocessed source article. 
- `year`: String. The (often approximate) publication year of the article. The exact year is often difficult to locate or scrape; in that case, we use (the author's year of birth + the author's year of death) / 2 as the approximate publication year. 
- `license`: String. The license information for the article. 
- `article`: String. The HTML of the article. A script that converts HTML to plain texts is provided. 
- `questions`: A list of dictionaries explained below. Each line of json has a different number of questions because some questions were removed following validation.

As discussed, the value of `questions` is a list of dictionaries. Each dictionary has the following fields.
- `question`: The question. 
- `options`: A list of four answer options.
- `gold_label`: The correct answer, defined by a majority vote of 3 or 5 annotators + the original writer's label. The number corresponds to the option number (1-indexed) in `options`. 
- `writer_label`: The label the writer provided. The number corresponds to the option number (1-indexed) in `options`. 
- `validation`: A list of dictionaries containing the untimed validation results. Each dictionary contains the following fields.
    - `untimed_annotator_id`: The anonymized annotator IDs corresponding to the untimed validation results shown in `untimed_answer`.
    - `untimed_answer`: The responses in the untimed validation. Each question in the training set is annotated by three workers in most cases, and each question in the dev/test sets is annotated by five cases in most cases (see paper for exceptions). 
    - `untimed_eval1_answerability`: The responses (represented numerically) to the first eval question in untimed validation. We asked the raters: “Is the question answerable and unambiguous?” The values correspond to the following choices:
        - 1: Yes, there is a single answer choice that is the most correct.
        - 2: No, two or more answer choices are equally correct.
        - 3: No, it is unclear what the question is asking, or the question or answer choices are unrelated to the passage.
    - `untimed_eval2_context`: The responses (represented numerically) to the second eval question in untimed validation. We asked the raters: “How much of the passage/text is needed as context to answer this question correctly?” The values correspond to the following choices:
        - 1: Only a sentence or two of context.
        - 2: At least a long paragraph or two of context.
        - 3: At least a third of the passage for context.
        - 4: Most or all of the passage for context.
    - `untimed_eval3_distractor`: The responses to the third eval question in untimed validation. We asked the raters: “Which of the options that you did not select was the best "distractor" item (i.e., an answer choice that you might be tempted to select if you hadn't read the text very closely)?” The numbers correspond to the option numbers (1-indexed).
- `speed_validation`: A list of dictionaries containing the speed validation results. Each dictionary contains the following fields.
    - `speed_annotator_id`: The anonymized annotator IDs corresponding to the speed annotation results shown in `speed_answer`.
    - `speed_answer`: The responses in the speed validation. Each question is annotated by five workers.
- `difficult`: A binary value. `1` means that less than 50% of the speed annotations answer the question correctly, so we include this question in the `hard` subset. Otherwise, the value is `0`. In our evaluations, we report one accuracy figure for the entire dataset, and a second for the `difficult=1` subset.

### Validation criteria for the questions
- More than 50% of annotators answer the question correctly in the untimed setting. That is, more than 50% of the `untimed_answer` annotations agree with `gold_label` (defined as the majority vote of validators' annotations together with the writer's provided label).
- More than 50% of annotators think that the question is unambiguous and answerable. That is, more than 50% of the `untimed_eval1_answerability` annotations have `1`'s.

### <a name="difficult">What are the `hard` questions?</a>
 - More than 50% of annotators answer the question correctly in the untimed setting. That is, more than 50% of the `untimed_answer` annotations agree with `gold_label`.
 - More than 50% of annotators think that the question is unambiguous and answerable. That is, more than 50% of the `untimed_eval1_answerability` annotations have `1`'s.
 - More than 50% of annotators answer the question incorrectly in the speed validaiton setting. That is, more than 50% of the `speed_answer` annotations are incorrect.

### Test set

The annotations for questions in the test set will not be released. [The leaderboard](https://nyu-mll.github.io/quality/) is up. Please submit!

## Code

The code for our baseline models is in `baselines`.

## Log

- June 2022: Our NAACL paper relies on QuALITY-v0.9. We discovered an issue in the HTML stripping script. When there is a piece of string like `Belgian<br/>politics` in the original article, the stripped version would become `Belgianpolitics` instead of `Belgian politics`. This issue affects around 15% of the examples (or around 0.5% of the tokens) in the HTML-stripped files of v0.9. The original HTML-formatted data remain unchanged, so `QuALITY.v1.0.[train/dev/test]` is the same as `QuALITY.v0.9.[train/dev/test]`. In `QuALITY-v1.0.htmlstripped.[train/dev/test]`, we fixed the issue caused by the `<br/>` tag.

## Acknowledgments

This project has benefited from financial support to SB by Eric and Wendy Schmidt (made by recommendation of the Schmidt Futures program), Samsung Research (under the project *Improving Deep Learning using Latent Structure*), Samsung Advanced Institute of Technology (under the project *Next Generation Deep Learning: From Pattern Recognition to AI*), and Apple. This material is based upon work supported by the National Science Foundation under Grant Nos. 1922658 and 2046556. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## Citation

```
@inproceedings{pang-etal-2022-quality,
    title = "{Q}u{ALITY}: Question Answering with Long Input Texts, Yes!",
    author = "Pang, Richard Yuanzhe  and
      Parrish, Alicia  and
      Joshi, Nitish  and
      Nangia, Nikita  and
      Phang, Jason  and
      Chen, Angelica  and
      Padmakumar, Vishakh  and
      Ma, Johnny  and
      Thompson, Jana  and
      He, He  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.391",
    pages = "5336--5358",
    abstract = "To enable building and testing models on long-document comprehension, we introduce QuALITY, a multiple-choice QA dataset with context passages in English that have an average length of about 5,000 tokens, much longer than typical current models can process. Unlike in prior work with passages, our questions are written and validated by contributors who have read the entire passage, rather than relying on summaries or excerpts. In addition, only half of the questions are answerable by annotators working under tight time constraints, indicating that skimming and simple search are not enough to consistently perform well. Our baseline models perform poorly on this task (55.4{\%}) and significantly lag behind human performance (93.5{\%}).",
}
```

## Contact

{yzpang, alicia.v.parrish}@nyu.edu
