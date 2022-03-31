from functools import partial

from mclt.datasets.huggingface import (
    AllegroReviewsDataModule,
    AmazonReviewsDataModule,
    PolEmo2InDataModule, MTSCDataModule, CyberbullyingDetectionDataModule,
    TweetsHateSpeechDetectionDataModule, HateSpeechPLDataModule, HateSpeech18DataModule,
    HateSpeechOffensiveDataModule, SemEval2018Task1DataModule, GoEmotionsDataModule,
    XNLIDataModule, PANXDataModule, CDSCEntailmentDataModule
)

DATASETS = {
    'tweets_hate_speech_detection:en': TweetsHateSpeechDetectionDataModule,
    'cyberbullying_detection:pl': CyberbullyingDetectionDataModule,
    'hate_speech_pl:pl': HateSpeechPLDataModule,
    'hate_speech18:en': HateSpeech18DataModule,
    'hate_speech_offensive:en': HateSpeechOffensiveDataModule,

    'semeval_2018:en': partial(SemEval2018Task1DataModule, 'english'),
    'semeval_2018:es': partial(SemEval2018Task1DataModule, 'spanish'),
    'semeval_2018:ar': partial(SemEval2018Task1DataModule, 'arabic'),
    'go_emotions:en': GoEmotionsDataModule,

    'polemo_in:pl': PolEmo2InDataModule,
    'polemo_out:pl': PolEmo2InDataModule,
    'allegro_reviews:pl': AllegroReviewsDataModule,
    'amazon_reviews:en': partial(AmazonReviewsDataModule, 'en'),
    'amazon_reviews:de': partial(AmazonReviewsDataModule, 'de'),
    'amazon_reviews:es': partial(AmazonReviewsDataModule, 'es'),
    'amazon_reviews:fr': partial(AmazonReviewsDataModule, 'fr'),

    'mtsc:pl': partial(MTSCDataModule, 'pl'),
    'mtsc:en': partial(MTSCDataModule, 'en'),
    'mtsc:sw': partial(MTSCDataModule, 'sw'),
    'mtsc:es': partial(MTSCDataModule, 'es'),
    'mtsc:so': partial(MTSCDataModule, 'so'),
    'mtsc:sk': partial(MTSCDataModule, 'sk'),
    'mtsc:se': partial(MTSCDataModule, 'se'),
    'mtsc:ru': partial(MTSCDataModule, 'ru'),
    'mtsc:pt': partial(MTSCDataModule, 'pt'),
    'mtsc:hu': partial(MTSCDataModule, 'hu'),
    'mtsc:de': partial(MTSCDataModule, 'de'),
    'mtsc:cr': partial(MTSCDataModule, 'cr'),
    'mtsc:bg': partial(MTSCDataModule, 'bg'),
    'mtsc:bo': partial(MTSCDataModule, 'bo'),
    'mtsc:al': partial(MTSCDataModule, 'al'),

    'xnli:en': partial(XNLIDataModule, 'en'),
    'xnli:fr': partial(XNLIDataModule, 'fr'),
    'xnli:de': partial(XNLIDataModule, 'de'),
    'xnli:es': partial(XNLIDataModule, 'es'),
    'xnli:bg': partial(XNLIDataModule, 'bg'),
    'cdsc-e': CDSCEntailmentDataModule,
}
