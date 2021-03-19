# RuSimpleSentEval (RSSE)

## Описание задачи
Задача упрощения тестов  (text simplification) предполагает несколько постановок, из которых мы выбираем самую популярную: упрощение на уровне предложений. В такой постановке задача заключается в том, чтобы из сложного предложения получить упрощенное. 
Критерии сложности предложения включают в себя наличие сложных грамматических конструкций, в том числе, причастных и деепричастных оборотов, подчиненных предложений, наличие редких и неоднозначных слов и т.д.

Большая часть вычислительных моделей упрощения текста используют нейросетевые seq2seq модели, которые обучаются по параллельным данным, то есть, по парам сложное предложение – простое предложение. Организаторы составили набор таких параллельных предложений и предоставят их участникам соревнования для обучения моделей. Поскольку для русского языка подобного набора данных не существовало, он был создан специально для соревнования. Основой обучающего набора данных послужили переведенные с английского материалы Википедии (English Wikipedia) и упрощенной Википедии (Simple English Wikipedia), подвергнутые дополнительной фильтрации. Вторая часть набора данных, которая будет использоваться для оценки и тестирования, составлена на краудсорсинговой платформе.

В качестве метрики качества будет использована мера SARI (System output Against References and against the Input sentence). 

## Таймлайн соревнования
* **15.02.2021-20.02.2021** -- публикация данных 
* **15.02.2021--18.03.2021** -- [первая фаза соревнования](https://competitions.codalab.org/competitions/29037#phases) (отправка результатов для public test).
**Вы находитесь здесь**
* **18.03.2021--20.03.2021**  -- [вторая фаза соревнования](https://competitions.codalab.org/competitions/29037#phases) (отправка результатов для private test).
* **22.03.2021** -- официальное завершение соревнования и подведение итогов.
* **30.03.2021** -- дедлайн по подачи статей по результатам соревнования.

## Данные

* Автоматически переведенный на русский  корпус WikiLarge [1]. Данные доступны по [ссылке](https://drive.google.com/drive/folders/1jfij3KuiRbO_XoLiquSBP2mZafzPhrsL). 
* Собранный организаторами на краудсорсинговой платформе корпус. Dev датасет (пары сложное –  простое предложение) доступен по [ссылке](https://github.com/dialogue-evaluation/RuSimpleSentEval/blob/main/dev_sents.csv). Сложные предложения из public test для первой фазы соревнования доступны по [ссылке](public_test_only.csv).  Сложные предложения из private test для второй фазы соревнования доступны по [ссылке](hidden_test_only.csv). 
* Большой корпус кластеров парафраз заголовков ParaPhraserPlus. Данные доступны по [ссылке](http://paraphraser.ru/download/). 

1. Zhang, X. and Lapata, M., 2017, September. Sentence Simplification with Deep Reinforcement Learning. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (pp. 584-594).

## Метрика 
Мы используем метрику SARI, конкретно в реализации библиотеки [EASSE](https://github.com/feralvam/easse).

## Базовое решение

В качестве бэйзлайна выступает претренированная модель multilingual BART (mBART) [2,3], дообученная на переведённом на русский корпусе WikiLarge. Для обучения использована библиотека FairSeq [4], в частности, в следующий пример:
https://github.com/pytorch/fairseq/tree/master/examples/mbart

Для обучения модели mBART необходимо:

1. Скачать предобученную mBART:

```
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.CC25.tar.gz
```

2. Установить токенизатор [SentencePiece](https://github.com/google/sentencepiece):

```
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig -v
```


3. Установить [FairSeq](https://github.com/pytorch/fairseq):

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

4. Перед началом обучения необходимо предобработать данные, распределив соответствующие исходные и упрощённые предложения по разным файлам. Каждая строка соответствует в точности одному предложению. После чего происходит токенизация исходных и упрощённых предложений: 

```
SPM=/path/to/sentencepiece/build/src/spm_encode
BPE_MODEL=/path/to/mbart/directory/sentence.bpe.model
DATA_DIR=/path/to/data
SRC=en
TGT=ru
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/train.${SRC} > ${DATA_DIR}/train.spm.${SRC} &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/train.${TGT} > ${DATA_DIR}/train.spm.${TGT} &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/valid.${SRC} > ${DATA_DIR}/valid.spm.${SRC} &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/valid.${TGT} > ${DATA_DIR}/valid.spm.${TGT} &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/test.${SRC} > ${DATA_DIR}/test.spm.${SRC} &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/test.${TGT} > ${DATA_DIR}/test.spm.${TGT} &
```
5. Предобработка данных с использованием словаря ($DICT) предобученной mBART:

```
PREPROCESSED_DATA_DIR=/directory/to/save/preprocessed/data
DICT=/path/to/downloaded/mbart/model/directory/dict.txt
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA_DIR}/train.spm \
  --validpref ${DATA_DIR}/valid.spm \
  --testpref ${DATA_DIR}/test.spm \
  --destdir ${PREPROCESSED_DATA_DIR} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
```

6. Обучение модели. Полный список параметров можно получить вызовом "fairseq-train --help":

```
PRETRAIN=/path/to/downloaded/mbart/model/directory/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
CUDA_VISIBLE_DEVICES=0,1,2,3
SAVE_DIR=/path/to/save/model/checkpoint
fairseq-train ${PREPROCESSED_DATA_DIR} \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 54725  \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --source-lang ${SRC} --target-lang ${TGT} \
  --batch-size 16 \
  --validate-interval 1 \
  --patience 3 \
  --max-epoch 25 \
  --save-interval 5 --keep-last-epochs 10 --keep-best-checkpoints 2 \
  --seed 42 --log-format simple --log-interval 500 \
  --restore-file ${PRETRAIN} \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --ddp-backend no_c10d \
  --langs $langs \
  --scoring bleu \
  --save-dir ${SAVE_DIR} > train_log.txt &
```

Результат обучения (чекпоинт обученной модели) может выступать в качестве претренированной модели для дальнейшего обучения. Например, возможно обучить модель на оригинальном английском WikiLarge, после чего изменить путь до предобученной модели (${PRETRAIN}) на путь до чекпоинта, находящегося в директории ${SAVE_DIR}. Возможный путь до предобученной модели будет выглядеть как "${SAVE_DIR}/checkpoint15.pt".  

7. Предсказание модели может быть получено следующим образом:

```
CUDA_VISIBLE_DEVICES=0
LANG=C.UTF-8 LC_ALL=C.UTF-8
fairseq-generate ${DATA_DIR} \
  --path ${SAVE_DIR}/checkpoint_best.pt \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  --source-lang src --target-lang dst \
  --bpe 'sentencepiece' --sentencepiece-model ${BPE_MODEL} \
  --sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs $langs > model_prediction.txt & 

cat model_prediction.txt | grep -P "^H" |sort -V |cut -f 3- > model_prediction.hyp
```

### Литература:
1. Lewis, Mike, et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.

2. Liu, Yinhan, et al. "Multilingual denoising pre-training for neural machine translation." Transactions of the Association for Computational Linguistics 8 (2020): 726-742.

3. Ott, Myle, et al. "fairseq: A Fast, Extensible Toolkit for Sequence Modeling." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations). 2019.

## Оценка качества

Для оценки качества упрощения текстов используется библиотека EASSE [5].

1. Клонирование репозитория EASSE:

```
git clone https://github.com/feralvam/easse
```

2. Замена скрипта оценки SARI на немного модифицированной. Модификация необходима для поддержки датасетов, в которых различные примеры содержат различное, а не фиксированное число эталонных упрощения.
```
git clone https://github.com/Andoree/sent_simplification.git
cp sent_simplification/sari.py $EASSE_DIR/easse
```

3. Установка EASSE:
```
cd $EASSE_DIR
pip install .
```

4. Приведение данных к формату, принимаемому EASSE, с помощью скрипта из [этого](https://github.com/Andoree/sent_simplification) репозитория:
```
refs_to_easse_format.py \
--input_path $PATH_TO_ANNOTATED_DATA_CSV \
--src_column $SOURCE_COLUMN_NAME \
--trg_column $TARGET_COLUMN_NAME \
--output_dataset_name $DATASET_NAME \
--output_dir $REFERENCES_DIR
```

5. Подсчёт SARI и BLEU:
```
easse evaluate \
--test_set custom \
--metrics bleu,sari \
--refs_sents_paths $REFERENCES_DIR/$DATASET_NAME.ref.0,$REFERENCES_DIR/$DATASET_NAME.ref.1,$REFERENCES_DIR/$DATASET_NAME.ref.2,$REFERENCES_DIR/$DATASET_NAME.ref.3,$REFERENCES_DIR/$DATASET_NAME.ref.4,$REFERENCES_DIR/$DATASET_NAME.ref.5,$REFERENCES_DIR/$DATASET_NAME.ref.6,$REFERENCES_DIR/$DATASET_NAME.ref.7,$REFERENCES_DIR/$DATASET_NAME.ref.8,$REFERENCES_DIR/$DATASET_NAME.ref.9,$REFERENCES_DIR/$DATASET_NAME.ref.10,$REFERENCES_DIR/$DATASET_NAME.ref.11,$REFERENCES_DIR/$DATASET_NAME.ref.12,$REFERENCES_DIR/$DATASET_NAME.ref.13,$REFERENCES_DIR/$DATASET_NAME.ref.14,$REFERENCES_DIR/$DATASET_NAME.ref.15,$REFERENCES_DIR/$DATASET_NAME.ref.16,$REFERENCES_DIR/$DATASET_NAME.ref.17,$REFERENCES_DIR/$DATASET_NAME.ref.18,$REFERENCES_DIR/$DATASET_NAME.ref.19,$REFERENCES_DIR/$DATASET_NAME.ref.20  \
-=orig_sents_path $REFERENCES_DIR/$DATASET_NAME.src \
--sys_sents_path $MODEL_PREDICTION_PATH -q
```


[5] Alva-Manchego, Fernando, et al. "EASSE: Easier Automatic Sentence Simplification Evaluation." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations. 2019.

## Организаторы:
* Екатерина Артемова, ВШЭ, HUAWEI
* Александра Ижевская, ВШЭ
* Валентин Малых, КФУ, HUAWEI
* Алена Пестова, ВШЭ
* Андрей Саховский, КФУ
* Иван Смуров, ABBYY, МФТИ
* Елена Тутубалина, КФУ, ВШЭ


[**Страница соревнования на CodaLab**](https://competitions.codalab.org/competitions/29037#learn_the_details)

[**Телеграм-чат соревнования**](https://t.me/rsse2021)
