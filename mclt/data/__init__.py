from functools import partial

import pandas as pd

from mclt.data.datamodules import (
    AllegroReviewsDataModule,
    AmazonReviewsDataModule,
    CDSCEntailmentDataModule,
    CEDRDataModule,
    CSFeverDataModule,
    CyberbullyingDetectionDataModule,
    GoEmotionsDataModule,
    HateSpeech18DataModule,
    HateSpeechOffensiveDataModule,
    HateSpeechPLDataModule,
    HatEvalDataModule,
    MallCZDataModule,
    MTSCDataModule,
    PANXDataModule,
    PolEmo2InDataModule,
    RuReviewsDataModule,
    RuUkAbusiveLanguageDataModule,
    SemEval2018Task1DataModule,
    TweetsHateSpeechDetectionDataModule,
    XEDDataModule,
    XNLIDataModule, CSFDDataModule,
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
    'mtsc:si': partial(MTSCDataModule, 'si'),
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
    'xnli:ru': partial(XNLIDataModule, 'ru'),
    'xnli:de': partial(XNLIDataModule, 'de'),
    'xnli:es': partial(XNLIDataModule, 'es'),
    'xnli:bg': partial(XNLIDataModule, 'bg'),
    'cdsc-e:pl': CDSCEntailmentDataModule,
    'cedr:ru': CEDRDataModule,
    'rureviews:ru': RuReviewsDataModule,
    'ru_uk_abusive_language:ru': RuUkAbusiveLanguageDataModule,
    'hateval:en': partial(HatEvalDataModule, 'en'),
    'hateval:es': partial(HatEvalDataModule, 'es'),
    'mallcz:cz': MallCZDataModule,
    'csfever:cz': CSFeverDataModule,
    'csfd:cz': CSFDDataModule,
    'xed:si': partial(XEDDataModule, 'si'),
    'xed:pl': partial(XEDDataModule, 'pl'),
    'xed:cz': partial(XEDDataModule, 'cz'),
}


TASK_LANGUAGE_TABLE = pd.DataFrame(
    [
        ['mtsc:pl', 'allegro_reviews:pl', 'cyberbullying_detection:pl', 'xed:pl', 'cdsc-e:pl'],
        ['csfd:cz', 'mallcz:cz', None, 'xed:cz', 'csfever:cz'],
        ['mtsc:si', None, None, 'xed:si', None],
        ['mtsc:en', 'amazon_reviews:en', 'hateval:en', 'semeval_2018:en', 'xnli:en'],
        ['mtsc:es', 'amazon_reviews:es', 'hateval:es', 'semeval_2018:es', 'xnli:es'],
        ['mtsc:ru', 'rureviews:ru', 'ru_uk_abusive_language:ru', 'cedr:ru', 'xnli:ru'],
    ],
    columns=['sentiment', 'reviews', 'hate-speech', 'emotions', 'nli'],
    index=['pl', 'cz', 'si', 'en', 'es', 'ru'],
)


DATASET_TO_TASK_LANG = {
    dataset: (lang, task)
    for lang, row in TASK_LANGUAGE_TABLE.iterrows()
    for task, dataset in row.iteritems()
}
