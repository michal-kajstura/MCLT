## Research questions
   1. How well do the multilingual architectures perform in various tasks 
   when trained on multiple languages?
      1. Select a task
      2. Train multilingual model on all available languages for the task
      3. Compare with monolingual baselines.
   2. If we trained a model on a particular language family,
   how well does the model perform on a language that belongs
   to the same family in comparison to a model trained on the
   entire set of languages?
      1. Same task as in 5.i.a
      2. Train only on similar languages
      3. Compare with monolingual and general-multilingual model
   3. Can we identify auxiliary tasks that benefit the target task? Sentiment analysis dataset
   could improve results for hate speech detection.
   Try this in monolingual setting (e.g. Polish and PolEmo + Cyberbullying)
      1. Same task as in 5.i.a
      2. Multi-task training
      3. Compare with cross-lingual transfer
   4. Cross lingual multi-task transfer. Pick a dataset D1 in language L1.
   Can we find a (D2, L2) that improves monolingual results?
      1. Low resource scenario
      2. E.g. **Automating News Comment Moderation with Limited Resources: Benchmarking in Croatian and Estonian**
      3. Add PolEval, Cyberbullying

## Language families:
Slavic: pl, sl, hr, sk, cs, bg, uk, ru, be
Baltic: lv, lt
Uralic: et, el, sr, fi, hu
Romance/Germanic: de, pt, fr, ro, en, es, ...

## Datasets
|     | Sentiment            | Reviews                         | Hate-speech                                                                                            | Emotions                | NLI     |
|-----|----------------------|---------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------|---------|
| pl  | MTSC                 | PolEmo 2.0  <br> AllegroReviews | HateSpeechPL <br> CyberbullyingDetection                                                               | XED                     | CDSC-E  |
| sk  | MTSC, Sentigrade     |                                 |                                                                                                        | XED                     |         |
| si  | MTSC, SentiNews      |                                 |                                                                                                        | XED                     |         |
| cz  | CSFD <br> FacebookCZ | MallCZ                          |                                                                                                        | XED                     | csfever |
| en  | MTSC                 | AmazonReviews                   | HateSpeech18 <br> HateSpeechOffensive <br> MeasuringHateSpeech <br> TweetsHateSpeech  <br> HatEval2019 | SemEval <br> GoEmotions |         |
| de  | MTSC                 | AmazonReviews                   | FBHateSpeech <br> rp_mod_crowd                                                                         | XED                     | XNLI    |
| fr  |                      | AmazonReviews                   | HateSpeechMLMA                                                                                         |                         | XNLI    |
| es  | MTSC                 | AmazonReviews                   | HaterSpeech <br> HatEval2019                                                                           | SemEval                 | XNLI    |
| ru  | MTSC                 | rureviews                       | AbusiveLanguageDataset                                                                           | cedr                    | XNLI    |
