# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e


#
# Data preprocessing configuration
#
N_THREADS=16    # number of threads in data preprocessing


#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --reload_codes)
    RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
    RELOAD_VOCAB="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


#
# Check parameters
#
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi
if [ "$TGT" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$SRC" != "de" -a "$SRC" != "en" -a "$SRC" != "fr" -a "$SRC" != "ro" ]; then echo "unknown source language"; exit; fi
if [ "$TGT" != "de" -a "$TGT" != "en" -a "$TGT" != "fr" -a "$TGT" != "ro" ]; then echo "unknown target language"; exit; fi
if [ "$SRC" == "$TGT" ]; then echo "source and target cannot be identical"; exit; fi
if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi


#
# Initialize tools and data paths
#

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $PROC_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# BPE / vocab files 
BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.$SRC
TGT_VOCAB=$PROC_PATH/vocab.$TGT
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT


# train / valid / test parallel BPE data
PARA_SRC_TRAIN_BPE=$PROC_PATH/train.$SRC-$TGT.$SRC  # Update
PARA_TGT_TRAIN_BPE=$PROC_PATH/train.$SRC-$TGT.$TGT  # Update
PARA_SRC_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$SRC
PARA_TGT_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$TGT
PARA_SRC_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$SRC
PARA_TGT_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$TGT

# valid / test file raw data
unset PARA_SRC_TRAIN PARA_TGT_TRAIN PARA_SRC_VALID PARA_TGT_VALID PARA_SRC_TEST PARA_TGT_TEST    # Update

if [ "$SRC" == "de" -a "$TGT" == "en" ]; then
  PARA_SRC_TRAIN_RAW=$PARA_PATH/train/raw/de-en-wmt14/train-raw.de  # Update  这是未经处理的数据
  PARA_TGT_TRAIN_RAW=$PARA_PATH/train/raw/de-en-wmt14/train-raw.en  # Update
  PARA_SRC_VALID_RAW=$PARA_PATH/dev/raw/valid-raw.de  # Update
  PARA_TGT_VALID_RAW=$PARA_PATH/dev/raw/valid-raw.en  # Update
  PARA_SRC_TEST_RAW=$PARA_PATH/dev/raw/test-raw.de  # Update
  PARA_TGT_TEST_RAW=$PARA_PATH/dev/raw/test-raw.en  # Update

  PARA_SRC_TRAIN=$PARA_PATH/train/raw/de-en-wmt14/train.de  # Update
  PARA_TGT_TRAIN=$PARA_PATH/train/raw/de-en-wmt14/train.en  # Update
  PARA_SRC_VALID=$PARA_PATH/dev/valid.de  # 用德语和英语来做验证集和测试集，所以不写de-en，不必要与fr-en区别开，因为没有fr-en的验证集
  PARA_TGT_VALID=$PARA_PATH/dev/valid.en
  PARA_SRC_TEST=$PARA_PATH/dev/test.de
  PARA_TGT_TEST=$PARA_PATH/dev/test.en
fi

if [ "$SRC" == "en" -a "$TGT" == "fr" ]; then
  PARA_SRC_TRAIN_RAW=$PARA_PATH/train/raw/fr-en-wmt14/train-raw.en  # Update  这是未经处理的数据
  PARA_TGT_TRAIN_RAW=$PARA_PATH/train/raw/fr-en-wmt14/train-raw.fr  # Update
  PARA_SRC_VALID_RAW=$PARA_PATH/train/raw/fr-en-wmt14/valid-raw.en  # Update
  PARA_TGT_VALID_RAW=$PARA_PATH/train/raw/fr-en-wmt14/valid-raw.fr  # Update
  PARA_SRC_TEST_RAW=$PARA_PATH/train/raw/fr-en-wmt14/test-raw.en  # Update
  PARA_TGT_TEST_RAW=$PARA_PATH/train/raw/fr-en-wmt14/test-raw.fr  # Update

  PARA_SRC_TRAIN=$PARA_PATH/train/raw/fr-en-wmt14/train.en  # Update
  PARA_TGT_TRAIN=$PARA_PATH/train/raw/fr-en-wmt14/train.fr  # Update
  PARA_SRC_VALID=$PARA_PATH/train/raw/fr-en-wmt14/valid.en
  PARA_TGT_VALID=$PARA_PATH/train/raw/fr-en-wmt14/valid.fr
  PARA_SRC_TEST=$PARA_PATH/train/raw/fr-en-wmt14/test.en
  PARA_TGT_TEST=$PARA_PATH/train/raw/fr-en-wmt14/test.fr
fi
# preprocessing commands - special case for Romanian
if [ "$SRC" == "ro" ]; then
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
else
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
fi
if [ "$TGT" == "ro" ]; then
  TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"
else
  TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"
fi

# cd $PARA_PATH


# check valid and test files are here
if ! [[ -f "$PARA_SRC_TRAIN_RAW" ]]; then echo "$PARA_SRC_TRAIN_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_TGT_TRAIN_RAW" ]]; then echo "$PARA_TGT_TRAIN_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_SRC_VALID_RAW" ]]; then echo "$PARA_SRC_VALID_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_TGT_VALID_RAW" ]]; then echo "$PARA_TGT_VALID_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_SRC_TEST_RAW" ]];  then echo "$PARA_SRC_TEST_RAW is not found!";  exit; fi
if ! [[ -f "$PARA_TGT_TEST_RAW" ]];  then echo "$PARA_TGT_TEST_RAW is not found!";  exit; fi


echo "Tokenizing train data..."
eval "cat $PARA_SRC_TRAIN_RAW | $SRC_PREPROCESSING > $PARA_SRC_TRAIN"
eval "cat $PARA_TGT_TRAIN_RAW | $TGT_PREPROCESSING > $PARA_TGT_TRAIN"

echo "Tokenizing valid and test data..."
eval "cat $PARA_SRC_VALID_RAW | $SRC_PREPROCESSING > $PARA_SRC_VALID"  # Update 下同
eval "cat $PARA_TGT_VALID_RAW | $TGT_PREPROCESSING > $PARA_TGT_VALID"
eval "cat $PARA_SRC_TEST_RAW | $SRC_PREPROCESSING > $PARA_SRC_TEST"
eval "cat $PARA_TGT_TEST_RAW | $TGT_PREPROCESSING > $PARA_TGT_TEST"

# reload BPE codes
cd $MAIN_PATH
if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $PARA_SRC_TRAIN $PARA_TGT_TRAIN > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

echo "Applying BPE to train files..."
$FASTBPE applybpe $PARA_SRC_TRAIN_BPE $PARA_SRC_TRAIN $BPE_CODES     # Update
$FASTBPE applybpe $PARA_TGT_TRAIN_BPE $PARA_TGT_TRAIN $BPE_CODES     # Update

# extract source and target vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $PARA_SRC_TRAIN_BPE > $SRC_VOCAB
  $FASTBPE getvocab $PARA_TGT_TRAIN_BPE > $TGT_VOCAB
fi
echo "$SRC vocab in: $SRC_VOCAB"
echo "$TGT vocab in: $TGT_VOCAB"

# reload full vocabulary
cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $PARA_SRC_TRAIN_BPE $PARA_TGT_TRAIN_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"


echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $PARA_SRC_VALID_BPE $PARA_SRC_VALID $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $PARA_TGT_VALID_BPE $PARA_TGT_VALID $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $PARA_SRC_TEST_BPE  $PARA_SRC_TEST  $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $PARA_TGT_TEST_BPE  $PARA_TGT_TEST  $BPE_CODES $TGT_VOCAB


echo "Binarizing data..."
rm -f $PARA_SRC_TRAIN_BPE.pth $PARA_TGT_TRAIN_BPE.pth $PARA_SRC_VALID_BPE.pth $PARA_TGT_VALID_BPE.pth $PARA_SRC_TEST_BPE.pth $PARA_TGT_TEST_BPE.pth     # Update
echo "Binarizing train data..."
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TRAIN_BPE    # Update
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TRAIN_BPE    # Update
echo "Binarizing test data..."
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TEST_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TEST_BPE

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Parallel training data:"
echo "    $SRC: $PARA_SRC_TRAIN_BPE.pth"
echo "    $TGT: $PARA_TGT_TRAIN_BPE.pth"
echo "Parallel validation data:"
echo "    $SRC: $PARA_SRC_VALID_BPE.pth"
echo "    $TGT: $PARA_TGT_VALID_BPE.pth"
echo "Parallel test data:"
echo "    $SRC: $PARA_SRC_TEST_BPE.pth"
echo "    $TGT: $PARA_TGT_TEST_BPE.pth"
echo ""

